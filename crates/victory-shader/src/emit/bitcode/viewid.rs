//! Which input scalars each output scalar depends on, as the DXIL
//! validator computes it (DXC's `ComputeViewIdStateBuilder`) to check the
//! `dx.viewIdState` metadata and `PSV0`'s dependency tables: the validator
//! wants the same sets, neither more nor fewer.
//!
//! An output scalar depends on the `loadInput` calls among everything its
//! stored values are made of, followed through:
//! - operands, transitively;
//! - for a load or an atomic, every store or atomic into the same global;
//! - the branches the block of every such instruction (and of the store
//!   itself) is control-dependent on, and their conditions;
//! - for a phi's constant operands, the branches the highest block that
//!   may still "define" that constant is control-dependent on.
//!
//! The analysis runs over the function as it is written: the blocks
//! [`Module::layout`] keeps, with folded latches replaced by their
//! predecessor, and without phis nothing reads.

use std::collections::HashMap;

use super::{Module, Op, Term, V, successors};

/// `dx.op` opcodes.
const LOAD_INPUT: i64 = 4;
const STORE_OUTPUT: i64 = 5;

/// A node of the dependence walk: an instruction or a block's terminator.
#[derive(Clone, Copy)]
enum Node {
    Inst(u32),
    Term(usize),
}

/// The written function's control flow.
struct Cfg {
    /// Written blocks in order (the entry first).
    order: Vec<usize>,
    /// For each folded latch, its predecessor and loop header.
    fold: Vec<Option<(usize, usize)>>,
    /// Immediate dominator (`None` for the entry and unwritten blocks).
    idom: Vec<Option<usize>>,
    /// Blocks each block is control-dependent on.
    cd: Vec<Vec<usize>>,
    /// For each constant, the first constant equal to it (LLVM keeps one
    /// of each).
    canon: Vec<u32>,
}

impl Module {
    /// For each input scalar (`row * 4 + column`), the output scalars that
    /// depend on it, as a mask (bit `row * 4 + column` of the output).
    /// `inputs` and `outputs` give each signature element's first scalar,
    /// `None` for an element without a register.
    pub(crate) fn view_dependencies(
        &self,
        inputs: &[Option<u32>],
        outputs: &[Option<u32>],
    ) -> Vec<u128> {
        let cfg = self.cfg();
        let live = self.live();
        // The block of every written instruction.
        let mut block_of = vec![usize::MAX; self.insts.len()];
        for &b in &cfg.order {
            let blk = &self.blocks[b];
            for &i in blk
                .phis
                .iter()
                .filter(|&&i| live[i as usize])
                .chain(&blk.body)
            {
                block_of[i as usize] = b;
            }
        }
        let written = |i: u32| block_of[i as usize] != usize::MAX;
        // Users of globals and instructions (the written ones).
        let mut users: HashMap<V, Vec<u32>> = HashMap::new();
        for (i, inst) in self.insts.iter().enumerate() {
            if !written(i as u32) {
                continue;
            }
            super::for_each_operand(&inst.op, |v| {
                if matches!(v, V::Inst(_) | V::Global(_)) {
                    users.entry(v).or_default().push(i as u32);
                }
            });
        }

        let dx_op = |i: u32| -> Option<(i64, &[V])> {
            let Op::Call(f, args) = &self.insts[i as usize].op else {
                return None;
            };
            if !self.funcs[*f as usize].name.starts_with("dx.op.") {
                return None;
            }
            Some((self.int_value(*args.first()?)?, &args[1..]))
        };
        let scalar = |bases: &[Option<u32>], args: &[V]| -> Option<u32> {
            let id = self.int_value(args[0])? as usize;
            let base = (*bases.get(id)?)?;
            // Elements are one row: a dynamic row index is row 0.
            let row = self.int_value(args[1]).unwrap_or(0) as u32;
            let col = self.int_value(args[2])? as u32;
            Some(base + row * 4 + col)
        };

        // Stores into each global, and the constant-phi branches of each
        // phi, worked out once.
        let mut stores: HashMap<V, Vec<u32>> = HashMap::new();
        let mut phi_blocks: HashMap<u32, Vec<usize>> = HashMap::new();

        // The stores to each output scalar.
        let mut by_output: Vec<(u32, Vec<u32>)> = Vec::new();
        for &b in &cfg.order {
            for &i in &self.blocks[b].body {
                if let Some((STORE_OUTPUT, args)) = dx_op(i)
                    && let Some(o) = scalar(outputs, args)
                {
                    match by_output.iter_mut().find(|(s, _)| *s == o) {
                        Some((_, v)) => v.push(i),
                        None => by_output.push((o, vec![i])),
                    }
                }
            }
        }

        let n = self.insts.len();
        let mut seen = vec![u32::MAX; n + self.blocks.len()];
        let mut deps: Vec<u128> = Vec::new();
        for (stamp, (o, calls)) in by_output.iter().enumerate() {
            let stamp = stamp as u32;
            let mut work: Vec<Node> = Vec::new();
            let mut visit = |node: Node, work: &mut Vec<Node>| {
                let k = match node {
                    Node::Inst(i) => i as usize,
                    Node::Term(b) => n + b,
                };
                if seen[k] != stamp {
                    seen[k] = stamp;
                    work.push(node);
                }
            };
            for &call in calls {
                let Op::Call(_, args) = &self.insts[call as usize].op else {
                    unreachable!()
                };
                if let V::Inst(v) = args[4]
                    && written(v)
                {
                    visit(Node::Inst(v), &mut work);
                }
                for &c in &cfg.cd[block_of[call as usize]] {
                    visit(Node::Term(c), &mut work);
                }
            }
            while let Some(node) = work.pop() {
                let block = match node {
                    Node::Term(b) => {
                        if let Some(Term::CondBr(V::Inst(c), ..)) = self.blocks[b].term
                            && written(c)
                        {
                            visit(Node::Inst(c), &mut work);
                        }
                        b
                    }
                    Node::Inst(i) => {
                        let op = &self.insts[i as usize].op;
                        match op {
                            Op::Phi(_) => {
                                let blocks = phi_blocks
                                    .entry(i)
                                    .or_insert_with(|| self.phi_blocks(i, &cfg, &block_of));
                                for &c in blocks.iter() {
                                    for &d in &cfg.cd[c] {
                                        visit(Node::Term(d), &mut work);
                                    }
                                }
                            }
                            Op::Load(p) | Op::AtomicRmw(_, p, _) | Op::CmpXchg(p, ..) => {
                                let mut decls = Vec::new();
                                self.reaching_decls(*p, &mut decls, &mut Vec::new());
                                for d in decls {
                                    let s = stores
                                        .entry(d)
                                        .or_insert_with(|| collect_stores(self, &users, d));
                                    for &s in s.iter() {
                                        visit(Node::Inst(s), &mut work);
                                    }
                                }
                            }
                            Op::Call(..) => {
                                if let Some((LOAD_INPUT, args)) = dx_op(i)
                                    && let Some(input) = scalar(inputs, args)
                                {
                                    let input = input as usize;
                                    if deps.len() <= input {
                                        deps.resize(input + 1, 0);
                                    }
                                    deps[input] |= 1 << o;
                                }
                            }
                            _ => {}
                        }
                        super::for_each_operand(op, |v| {
                            if let V::Inst(v) = v
                                && written(v)
                            {
                                visit(Node::Inst(v), &mut work);
                            }
                        });
                        block_of[i as usize]
                    }
                };
                for &c in &cfg.cd[block] {
                    visit(Node::Term(c), &mut work);
                }
            }
        }
        deps
    }

    /// The written control flow, its dominators and control dependence.
    fn cfg(&self) -> Cfg {
        let (order, fold) = self.layout();
        let nb = self.blocks.len();
        let mut pos = vec![usize::MAX; nb];
        for (k, &b) in order.iter().enumerate() {
            pos[b] = k;
        }
        let mut succs = vec![Vec::new(); nb];
        let mut preds = vec![Vec::new(); nb];
        for &b in &order {
            for &t in successors(self.blocks[b].term).iter() {
                let t = fold[t].map_or(t, |(_, h)| h);
                succs[b].push(t);
                preds[t].push(b);
            }
        }
        // Dominators (Cooper, Harvey and Kennedy), over the reverse
        // postorder the layout is.
        let idom = dominators(&order, &preds, &pos);
        // Post-dominators: the same over the reversed graph from a virtual
        // exit (index `nb`) every returning block leads to.
        let mut rpreds = vec![Vec::new(); nb + 1];
        let mut rorder = Vec::new();
        {
            let mut seen = vec![false; nb + 1];
            let mut post = Vec::new();
            let mut rsuccs = vec![Vec::new(); nb + 1];
            for &b in &order {
                if succs[b].is_empty() {
                    rsuccs[nb].push(b);
                    rpreds[b].push(nb);
                }
                for &t in &succs[b] {
                    rsuccs[t].push(b);
                    rpreds[b].push(t);
                }
            }
            let mut stack = vec![(nb, 0usize)];
            seen[nb] = true;
            while let Some(top) = stack.last_mut() {
                let (b, k) = *top;
                top.1 += 1;
                match rsuccs[b].get(k) {
                    Some(&s) if !seen[s] => {
                        seen[s] = true;
                        stack.push((s, 0));
                    }
                    Some(_) => {}
                    None => {
                        post.push(b);
                        stack.pop();
                    }
                }
            }
            post.reverse();
            rorder.extend(post);
        }
        let mut rpos = vec![usize::MAX; nb + 1];
        for (k, &b) in rorder.iter().enumerate() {
            rpos[b] = k;
        }
        let ipdom = dominators(&rorder, &rpreds, &rpos);
        // The virtual exit is no block: a block it immediately
        // post-dominates has none.
        let ipdom = |b: usize| ipdom[b].filter(|&p| p != nb);
        // x is control-dependent on y for every edge y -> s and every x on
        // the post-dominator path from s up to (not including) ipdom(y).
        let mut cd: Vec<Vec<usize>> = vec![Vec::new(); nb];
        for &y in &order {
            let stop = ipdom(y);
            for &s in &succs[y] {
                let mut x = Some(s);
                while let Some(b) = x {
                    if Some(b) == stop {
                        break;
                    }
                    if !cd[b].contains(&y) {
                        cd[b].push(y);
                    }
                    x = ipdom(b);
                }
            }
        }
        let mut first: HashMap<&(super::Ty, super::Const), u32> = HashMap::new();
        let canon = (0..self.consts.len() as u32)
            .map(|c| *first.entry(&self.consts[c as usize]).or_insert(c))
            .collect();
        Cfg {
            order,
            fold,
            idom,
            cd,
            canon,
        }
    }

    /// Where a global, or the declaration a pointer comes from, is.
    fn reaching_decls(&self, p: V, decls: &mut Vec<V>, seen: &mut Vec<V>) {
        if seen.contains(&p) {
            return;
        }
        seen.push(p);
        match p {
            V::Global(_) => {
                if !decls.contains(&p) {
                    decls.push(p);
                }
            }
            V::Inst(i) => match &self.insts[i as usize].op {
                Op::Gep(_, q, _) | Op::Cast(_, q) => self.reaching_decls(*q, decls, seen),
                Op::Phi(inc) => {
                    for &(q, _) in inc {
                        self.reaching_decls(q, decls, seen);
                    }
                }
                Op::Select(_, t, f) => {
                    self.reaching_decls(*t, decls, seen);
                    self.reaching_decls(*f, decls, seen);
                }
                _ => {
                    if !decls.contains(&p) {
                        decls.push(p);
                    }
                }
            },
            _ => {}
        }
    }

    /// The blocks whose control dependence a phi's constant operands add:
    /// for each, the highest dominator of its incoming block from which
    /// only that constant reaches the phi.
    fn phi_blocks(&self, phi: u32, cfg: &Cfg, block_of: &[usize]) -> Vec<usize> {
        let Op::Phi(inc) = &self.insts[phi as usize].op else {
            return Vec::new();
        };
        // Incoming blocks as written: a folded latch is its predecessor.
        let from = |b: usize| cfg.fold[b].map_or(b, |(p, _)| p);
        let canon = |v: V| match v {
            V::Const(c) => V::Const(cfg.canon[c as usize]),
            v => v,
        };
        let inc: Vec<(V, usize)> = inc.iter().map(|&(v, b)| (canon(v), from(b))).collect();
        let is_const = |v: V| !matches!(v, V::Inst(_));
        let entry = cfg.order[0];
        // Marker per dominator-tree node: the value defined there, or None
        // where two values meet.
        let mut markers: HashMap<usize, Option<V>> = HashMap::new();
        for &(v, b) in &inc {
            markers.entry(b).or_insert(Some(v));
        }
        for &(v, b) in &inc {
            let def = match v {
                V::Inst(i) => block_of[i as usize],
                _ => entry,
            };
            if b == def {
                continue;
            }
            let mut node = cfg.idom[b];
            while let Some(d) = node {
                match markers.get_mut(&d) {
                    None => {
                        markers.insert(d, Some(v));
                    }
                    Some(m) => {
                        if let Some(old) = *m
                            && old != v
                            && !(is_const(old) && is_const(v))
                        {
                            *m = None;
                        }
                        break;
                    }
                }
                if d == def {
                    break;
                }
                node = cfg.idom[d];
            }
        }
        let mut out = Vec::new();
        for &(v, b) in &inc {
            if !is_const(v) {
                continue;
            }
            let (mut node, mut def) = (Some(b), b);
            while let Some(d) = node {
                if markers.get(&d) != Some(&Some(v)) {
                    break;
                }
                def = d;
                node = cfg.idom[d];
            }
            if !out.contains(&def) {
                out.push(def);
            }
        }
        out
    }
}

/// The stores and atomics into the declaration `d`: its users, through
/// everything but loads.
fn collect_stores(m: &Module, users: &HashMap<V, Vec<u32>>, d: V) -> Vec<u32> {
    let mut out = Vec::new();
    let mut seen = vec![d];
    let mut work = vec![d];
    while let Some(v) = work.pop() {
        for &u in users.get(&v).map_or(&[][..], |u| &u[..]) {
            let uv = V::Inst(u);
            if seen.contains(&uv) {
                continue;
            }
            seen.push(uv);
            match m.insts[u as usize].op {
                Op::Load(_) => {}
                Op::Store(..) | Op::AtomicRmw(..) | Op::CmpXchg(..) => out.push(u),
                _ => work.push(uv),
            }
        }
    }
    out
}

/// Immediate dominators by Cooper, Harvey and Kennedy's iteration over
/// `order` (a reverse postorder from its first block), with `preds` and
/// each block's position `pos` in `order`.
fn dominators(order: &[usize], preds: &[Vec<usize>], pos: &[usize]) -> Vec<Option<usize>> {
    let mut idom: Vec<Option<usize>> = vec![None; preds.len()];
    let Some(&root) = order.first() else {
        return idom;
    };
    idom[root] = Some(root);
    let intersect = |idom: &[Option<usize>], mut a: usize, mut b: usize| {
        while a != b {
            while pos[a] > pos[b] {
                a = idom[a].unwrap();
            }
            while pos[b] > pos[a] {
                b = idom[b].unwrap();
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &order[1..] {
            let mut new = None;
            for &p in &preds[b] {
                if pos[p] == usize::MAX || idom[p].is_none() {
                    continue;
                }
                new = Some(match new {
                    None => p,
                    Some(n) => intersect(&idom, p, n),
                });
            }
            if new.is_some() && idom[b] != new {
                idom[b] = new;
                changed = true;
            }
        }
    }
    idom[root] = None;
    idom
}

#[cfg(test)]
mod tests {
    use super::super::super::container::SigElement;
    use super::super::super::shader::Shader;
    use super::super::{Attr, Term, pred};
    use crate::dxbc::{Container, FourCC};

    fn element(name: &str, kind: u8, row: u32, interp: u8) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp,
            row: Some(row),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        }
    }

    /// A vertex shader that copies the position, and writes TEXCOORD.x
    /// behind a branch on input TEXCOORD.x and TEXCOORD.y from a phi of two
    /// constants that branch selects: those two depend on TEXCOORD.x by
    /// control alone, the position's scalars each on its own input. The
    /// validator, where there is one, agrees.
    #[test]
    fn control_flow_and_phis_of_constants_are_dependencies() {
        let mut s = Shader::new(false);
        s.inputs = vec![element("POSITION", 0, 0, 0), element("TEXCOORD", 0, 1, 0)];
        s.outputs = vec![
            element("SV_Position", 3, 0, 4),
            element("TEXCOORD", 0, 1, 2),
        ];
        let (entry, then, merge) = (s.entry, s.m.new_block(), s.m.new_block());
        let load = |s: &mut Shader, b: usize, id: u32, col: u8| {
            let args = [s.m.i32(id), s.m.i32(0), s.m.i8(col), s.m.undef(s.t.i32)];
            s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args)
        };
        let store = |s: &mut Shader, b: usize, id: u32, col: u8, v| {
            let args = [s.m.i32(id), s.m.i32(0), s.m.i8(col), v];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        };
        let x = load(&mut s, entry, 1, 0);
        let zero = s.m.f32(0);
        let c = s.m.cmp(entry, pred::OGT, x, zero);
        s.m.terminate(entry, Term::CondBr(c, then, merge));
        let one = s.m.f32(1.0f32.to_bits());
        store(&mut s, then, 1, 0, one);
        s.m.terminate(then, Term::Br(merge));
        let phi = s.m.phi(merge, s.t.f32);
        s.m.set_phi(phi, vec![(zero, entry), (one, then)]);
        store(&mut s, merge, 1, 1, phi);
        for col in 0..4 {
            let v = load(&mut s, merge, 0, col);
            store(&mut s, merge, 0, col, v);
        }
        s.m.terminate(merge, Term::Ret);

        let deps =
            s.m.view_dependencies(&[Some(0), Some(4)], &[Some(0), Some(4)]);
        let want: Vec<u128> = vec![1, 2, 4, 8, 0b11_0000];
        assert_eq!(deps, want);

        let mut dxil = s.finish();
        let c = Container::parse(&dxil).unwrap();
        let psv = c.part(FourCC::PSV0).unwrap().data;
        // The last eight words: the table, one word per input scalar.
        let table: Vec<u32> = psv[psv.len() - 32..]
            .chunks(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect();
        assert_eq!(table, [1, 2, 4, 8, 0b11_0000, 0, 0, 0]);
        #[cfg(windows)]
        if let Ok(mut v) = super::super::super::validator::Validator::load(
            super::super::super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut dxil).unwrap();
        }
    }
}
