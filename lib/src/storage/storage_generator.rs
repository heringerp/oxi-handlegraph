#![allow(clippy::same_name_method)]
use super::numeric_encoder::{StrHash, StrLookup};
use super::{ChainedDecodingQuadIterator, Storage};
use crate::model::vocab::rdf;
use crate::model::{NamedNodeRef, Term};
use crate::storage::binary_encoder::QuadEncoding;
pub use crate::storage::error::StorageError;
use crate::storage::numeric_encoder::Decoder;
#[cfg(not(target_family = "wasm"))]
use crate::storage::numeric_encoder::{EncodedQuad, EncodedTerm};
use crate::storage::vg_vocab::{faldo, vg};
use crate::storage::DecodingQuadIterator;
use core::panic;
use genawaiter::{
    rc::{gen, Gen},
    yield_,
};
use gfa::gfa::Orientation;
use handlegraph::handle::{Direction, Handle};
use handlegraph::packed::PackedElement;
use handlegraph::packedgraph::index::OneBasedIndex;
use handlegraph::packedgraph::paths::StepPtr;
use handlegraph::pathhandlegraph::{path::PathStep, GraphPathsRef, IntoPathIds, PathBase};
use handlegraph::pathhandlegraph::{
    GraphPathNames, GraphPaths, GraphPathsSteps, PathId, PathSequences,
};
use handlegraph::{
    handlegraph::IntoHandles, handlegraph::IntoNeighbors, handlegraph::IntoSequences,
};
use oxrdf::vocab::rdfs;
use oxrdf::{Literal, NamedNode};
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::rc::Rc;
use std::vec::IntoIter;
use std::{iter, str};
use urlencoding::{decode, encode};

const LEN_OF_PATH_AND_SLASH: usize = 5;
const FIRST_RANK: u64 = 1;
const FIRST_POS: u64 = 1;

pub struct StorageGenerator {
    storage: Rc<Storage>,
}

impl StorageGenerator {
    pub fn new(storage: Storage) -> Self {
        Self {
            storage: Rc::new(storage),
        }
    }

    pub fn quads_for_pattern(
        &self,
        subject: Option<&EncodedTerm>,
        predicate: Option<&EncodedTerm>,
        object: Option<&EncodedTerm>,
        graph_name: &EncodedTerm,
    ) -> ChainedDecodingQuadIterator {
        let iter = GraphIter::new(self.storage.clone(), subject, predicate, object, graph_name);
        ChainedDecodingQuadIterator {
            first: DecodingQuadIterator {
                terms: Box::new(iter),
                encoding: QuadEncoding::Spog,
            },
            second: None,
        }
    }

    fn print_quad(&self, quad: &EncodedQuad) {
        let sub = match &quad.subject {
            EncodedTerm::NamedNode { iri_id: _, value } => value.clone(),
            _ => "NOT NAMED".to_owned(),
        };
        let pre = match &quad.predicate {
            EncodedTerm::NamedNode { iri_id: _, value } => value.clone(),
            _ => "NOT NAMED".to_owned(),
        };
        let obj = match &quad.object {
            EncodedTerm::NamedNode { iri_id: _, value } => value.clone(),
            EncodedTerm::SmallStringLiteral(value) => format!("\"{}\"", value).to_string(),
            EncodedTerm::IntegerLiteral(value) => value.to_string(),
            _ => "NOT NAMED".to_owned(),
        };
        println!("\t- {}\t{}\t{} .", sub, pre, obj);
    }

    fn term_to_text(&self, term: Option<&EncodedTerm>) -> String {
        if term.is_none() {
            return "None".to_owned();
        }
        match term.expect("Term is not none") {
            EncodedTerm::NamedNode { iri_id: _, value } => value.clone(),
            EncodedTerm::SmallStringLiteral(value) => format!("\"{}\"", value).to_string(),
            EncodedTerm::IntegerLiteral(value) => value.to_string(),
            _ => "NOT NAMED".to_owned(),
        }
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn get_str(&self, _key: &StrHash) -> Result<Option<String>, StorageError> {
        Ok(None)
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn contains_str(&self, _key: &StrHash) -> Result<bool, StorageError> {
        Ok(true)
    }
}

impl StrLookup for StorageGenerator {
    fn get_str(&self, key: &StrHash) -> Result<Option<String>, StorageError> {
        self.get_str(key)
    }

    fn contains_str(&self, key: &StrHash) -> Result<bool, StorageError> {
        self.contains_str(key)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IterMode {
    Uninitialized,
    Invalid,
    Single,
    All,
    Finished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubMode {
    Start,
    SingleNode(NodeState),
    AllNodes(NodeState),
    Step(StepState),
    Path,
    PathSteps,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeState {
    Type,
    Value,
    Edges,
    EdgesDirectional,
    Finished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StepState {
    TypeStep,
    TypeRegion,
    Node,
    NodeReverse,
    Rank,
    Position,
    Path,
    Begin,
    End,
    FaldoBegin(FaldoState),
    FaldoEnd(FaldoState),
    Finished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FaldoState {
    Positon,
    Exact,
    Type,
    Reference,
    Finished,
}

#[derive(Debug)]
struct StepInfos(StepPtr, u64, u64);

struct GraphIter {
    storage: Rc<Storage>,
    subject: Option<EncodedTerm>,
    predicate: Option<EncodedTerm>,
    object: Option<EncodedTerm>,
    graph_name: EncodedTerm,
    mode: IterMode,
    sub_mode: SubMode,
    handles: IntoIter<Handle>,
    curr_handle: Option<Handle>,
    edges: IntoIter<Handle>,
    path_ids: IntoIter<PathId>,
    curr_path: Option<PathId>,
    step: Option<StepInfos>,
}

impl Iterator for GraphIter {
    type Item = EncodedQuad;

    fn next(&mut self) -> Option<EncodedQuad> {
        //println!("\nnext state: {:?}, {:?}", self.mode, self.sub_mode);
        match self.mode {
            IterMode::Uninitialized => None,
            IterMode::Invalid => None,
            IterMode::Finished => None,
            IterMode::All => match self.sub_mode {
                SubMode::Start => {
                    if self.subject.is_some() {
                        self.sub_mode = SubMode::SingleNode(NodeState::Type);
                    } else {
                        self.handles = self.storage.graph.handles().collect::<Vec<_>>().into_iter();
                        self.sub_mode = SubMode::AllNodes(NodeState::Type);
                    }
                    None
                }
                SubMode::AllNodes(_) => self.nodes().or_else(|| {
                    self.sub_mode = SubMode::Path;
                    self.set_paths();
                    self.next()
                }),
                SubMode::Path => self.paths().or_else(|| {
                    self.sub_mode = SubMode::Step(StepState::TypeStep);
                    self.set_paths();
                    println!(
                        "Setting path for steps: {:?}, {:?}",
                        self.curr_path, self.step
                    );
                    self.set_first_step();
                    self.next()
                }),
                SubMode::Step(_) => self.steps(),
                _ => None,
            },
            IterMode::Single => match self.sub_mode {
                SubMode::Path => self.paths(),
                SubMode::Step(_) => self.steps(),
                SubMode::AllNodes(_) => self.nodes(),
                SubMode::SingleNode(_) => self.nodes(),
                SubMode::PathSteps => self.path_steps(),
                _ => panic!("Should never be called without setting submode"),
            },
        }
    }
}

impl GraphIter {
    pub fn new(
        storage: Rc<Storage>,
        subject: Option<&EncodedTerm>,
        predicate: Option<&EncodedTerm>,
        object: Option<&EncodedTerm>,
        graph_name: &EncodedTerm,
    ) -> Self {
        let mut result = Self {
            storage,
            subject: subject.map(|s| s.to_owned()),
            predicate: predicate.map(|p| p.to_owned()),
            object: object.map(|o| o.to_owned()),
            graph_name: graph_name.to_owned(),
            mode: IterMode::Uninitialized,
            sub_mode: SubMode::Start,
            handles: Vec::new().into_iter(),
            curr_handle: None,
            edges: Vec::new().into_iter(),
            path_ids: Vec::new().into_iter(),
            curr_path: None,
            step: None,
        };
        //result.iter = result.clone().quads_for_pattern();
        result.quads_for_pattern();
        // println!("Iter: {:?}, {:?}, {:?}", subject, predicate, object);
        // println!("Set state: {:?}, {:?}", result.mode, result.sub_mode);
        result
    }

    fn quads_for_pattern(&mut self) {
        // There should be no blank nodes in the data
        if self.subject.as_ref().is_some_and(|s| s.is_blank_node())
            || self.object.as_ref().is_some_and(|o| o.is_blank_node())
        {
            // println!("OF: blanks");
            self.mode = IterMode::Invalid;
        } else if self.is_vocab(self.predicate.as_ref(), rdf::TYPE) && self.object.is_some() {
            self.mode = IterMode::Single;
            self.type_triples();
        } else if self.is_vocab(self.predicate.as_ref(), vg::PATH_PRED) && self.object.is_some() {
            println!("Short");
            self.mode = IterMode::Single;
            self.set_paths();
            self.set_first_step();
            self.sub_mode = SubMode::PathSteps;
        } else if self.is_node_related() {
            // println!("OF: nodes");
            if self.subject.is_some() {
                self.mode = IterMode::Single;
                self.sub_mode = SubMode::SingleNode(NodeState::Type);
            } else {
                self.mode = IterMode::Single;
                self.set_nodes();
                self.sub_mode = SubMode::AllNodes(NodeState::Type);
            }
        } else if self.is_step_associated() {
            // println!("OF: steps");
            self.mode = IterMode::Single;
            self.set_paths();
            self.set_first_step();
            self.sub_mode = SubMode::Step(StepState::TypeStep);
        } else if self.is_vocab(self.predicate.as_ref(), rdfs::LABEL) {
            // println!("OF: rdfs::label");
            self.mode = IterMode::Single;
            self.set_paths();
            self.set_first_step();
            self.sub_mode = SubMode::Step(StepState::TypeStep);
        } else if self.subject.is_none() && self.predicate.is_none() && self.object.is_none() {
            // println!("OF: triple none");
            self.mode = IterMode::All;
            self.set_nodes();
            self.sub_mode = SubMode::AllNodes(NodeState::Type);
        } else if self.subject.is_some() {
            self.mode = IterMode::Single;
            self.sub_mode = match self.get_term_type(self.subject.as_ref().unwrap()) {
                Some(SubjectType::NodeIri) => SubMode::SingleNode(NodeState::Type), // TODO: is this correct?
                Some(SubjectType::PathIri) => {
                    self.set_paths();
                    SubMode::Path
                }
                Some(SubjectType::StepIri) => {
                    // println!("Doing step");
                    self.set_paths();
                    self.set_first_step();
                    SubMode::Step(StepState::TypeStep)
                }
                Some(SubjectType::StepBorderIri) => {
                    self.set_paths();
                    self.set_first_step();
                    SubMode::Step(StepState::TypeStep)
                }
                None => SubMode::Start,
            };
        } else {
            self.mode = IterMode::Invalid;
        }
    }

    fn set_nodes(&mut self) {
        self.handles = self.storage.graph.handles().collect::<Vec<_>>().into_iter();
        self.curr_handle = self.handles.next();
    }

    fn get_term_type(&self, term: &EncodedTerm) -> Option<SubjectType> {
        if let EncodedTerm::NamedNode { iri_id: _, value } = term {
            let mut parts = value.split("/").collect::<Vec<_>>();
            parts.reverse();
            if parts[1] == "node" {
                Some(SubjectType::NodeIri)
            } else if parts.contains(&"path") && parts[1] == "step" {
                Some(SubjectType::StepIri)
            } else if parts.contains(&"path") && parts[1] == "position" {
                Some(SubjectType::StepBorderIri)
            } else if parts.contains(&"path") {
                Some(SubjectType::PathIri)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn type_triples(&mut self) {
        let sm = if self.is_vocab(self.object.as_ref(), vg::NODE) {
            match self.subject {
                Some(_) => SubMode::SingleNode(NodeState::Type),
                None => {
                    self.set_nodes();
                    SubMode::AllNodes(NodeState::Type)
                }
            }
        } else if self.is_vocab(self.object.as_ref(), vg::PATH) {
            self.set_paths();
            SubMode::Path
        } else if self.is_step_associated_type() {
            self.set_paths();
            self.set_first_step();
            SubMode::Step(StepState::TypeStep)
        } else {
            panic!("There should always be a type submode!");
        };
        self.sub_mode = sm;
    }

    fn path_steps(&mut self) -> Option<EncodedQuad> {
        if let Some(path_id) = self.curr_path {
            if let Some(StepInfos(step, rank, _)) = self.step {
                let path_name = self.get_path_name(path_id).unwrap();
                let path_node = self.path_to_namednode(&path_name);
                let step_node = self.step_to_namednode(&path_name, rank);
                println!("{} - PATH - {}", rank, &path_name);
                if let Some(next_step) = self.storage.graph.path_next_step(path_id, step) {
                    self.step = Some(StepInfos(next_step, rank + 1, 3));
                } else {
                    self.step = None;
                    self.mode = IterMode::Finished;
                }
                return Some(EncodedQuad {
                    subject: step_node.unwrap(),
                    predicate: vg::PATH_PRED.into(),
                    object: path_node.unwrap(),
                    graph_name: self.graph_name.clone(),
                });
            } else {
                panic!("ps2");
            }
        } else {
            panic!("ps1");
        }
    }

    fn nodes(&mut self) -> Option<EncodedQuad> {
        let (triple, sm) = match self.sub_mode {
            SubMode::SingleNode(nts) => {
                let handle = Handle::new(
                    self.get_node_id(self.subject.as_ref().unwrap())
                        .expect("Subject is node"),
                    Orientation::Forward,
                );
                let (triple, nnts) = self.node_triple(handle, nts);
                (triple, SubMode::SingleNode(nnts))
            }
            SubMode::AllNodes(nts) => {
                if let Some(handle) = self.curr_handle {
                    if let (Some(triple), nnts) = self.node_triple(handle, nts) {
                        (Some(triple), SubMode::AllNodes(nnts))
                    } else {
                        self.curr_handle = self.handles.next();
                        self.sub_mode = SubMode::AllNodes(NodeState::Type);
                        (self.nodes(), self.sub_mode)
                    }
                } else {
                    (None, SubMode::AllNodes(NodeState::Type))
                }
            }
            _ => (None, SubMode::Start),
        };
        self.sub_mode = sm;
        triple
    }

    fn node_triple(&mut self, handle: Handle, nts: NodeState) -> (Option<EncodedQuad>, NodeState) {
        let sub = self.handle_to_namednode(handle).expect("Should be fine");
        let (triple, nnts) = match nts {
            NodeState::Type => (self.get_type_triple(sub), NodeState::Value),
            NodeState::Value => {
                self.edges = self
                    .storage
                    .graph
                    .neighbors(handle, Direction::Right)
                    .collect::<Vec<_>>()
                    .into_iter();
                (self.handle_to_triples(handle), NodeState::Edges)
            }
            NodeState::Edges => {
                if let Some(triple) = self.handle_to_edge_triples(handle, false) {
                    (Some(triple), NodeState::Edges)
                } else {
                    self.edges = self
                        .storage
                        .graph
                        .neighbors(handle, Direction::Right)
                        .collect::<Vec<_>>()
                        .into_iter();
                    (
                        self.handle_to_edge_triples(handle, true),
                        NodeState::EdgesDirectional,
                    )
                }
            }
            NodeState::EdgesDirectional => {
                if let Some(triple) = self.handle_to_edge_triples(handle, true) {
                    (Some(triple), NodeState::EdgesDirectional)
                } else {
                    (None, NodeState::Finished)
                }
            }
            NodeState::Finished => (None, NodeState::Finished),
        };
        if triple.is_none() && nts != NodeState::Finished {
            self.node_triple(handle, nnts)
        } else {
            (triple, nnts)
        }
    }

    fn get_type_triple(&self, sub: EncodedTerm) -> Option<EncodedQuad> {
        if self.is_vocab(self.predicate.as_ref(), rdf::TYPE)
            && (self.is_vocab(self.object.as_ref(), vg::NODE) || self.object.is_none())
        {
            // println!("NF: type self.predicate");
            Some(EncodedQuad::new(
                sub,
                rdf::TYPE.into(),
                vg::NODE.into(),
                self.graph_name.clone(),
            ))
        } else if self.predicate.is_none() && self.is_vocab(self.object.as_ref(), vg::NODE) {
            // println!("NF: node self.object");
            Some(EncodedQuad::new(
                sub,
                rdf::TYPE.into(),
                vg::NODE.into(),
                self.graph_name.clone(),
            ))
        } else if self.predicate.is_none() {
            // println!("NF: none self.predicate");
            Some(EncodedQuad::new(
                sub,
                rdf::TYPE.into(),
                vg::NODE.into(),
                self.graph_name.clone(),
            ))
        } else {
            None
        }
    }

    fn paths(&mut self) -> Option<EncodedQuad> {
        if let Some(path_id) = self.curr_path {
            self.curr_path = self.path_ids.next();
            let Some(path_name) = self.storage.graph.get_path_name(path_id) else {
                return self.paths();
            };
            let path_name = path_name.collect::<Vec<_>>();
            let path_name = str::from_utf8(&path_name).unwrap();
            let path_node = self.path_to_namednode(path_name);
            if self.subject.is_none() || path_node == self.subject {
                if (self.predicate.is_none() || self.is_vocab(self.predicate.as_ref(), rdf::TYPE))
                    && (self.object.is_none() || self.is_vocab(self.object.as_ref(), vg::PATH))
                {
                    return Some(EncodedQuad::new(
                        path_node.unwrap(),
                        rdf::TYPE.into(),
                        vg::PATH.into(),
                        self.graph_name.clone(),
                    ));
                }
            }
            self.paths()
        } else {
            None
        }
    }

    fn set_paths(&mut self) {
        self.path_ids = self
            .storage
            .graph
            .path_ids()
            .collect::<Vec<_>>()
            .into_iter();
        self.curr_path = self.path_ids.next();
    }

    fn set_first_step(&mut self) {
        if self.subject.is_none() {
            if let Some(path_id) = self.curr_path {
                let step = self
                    .storage
                    .graph
                    .path_first_step(path_id)
                    .expect("Every path should have at least one step");
                self.step = Some(StepInfos(step, FIRST_RANK, FIRST_POS));
            }
        } else if let Some(step_type) = self.get_step_iri_fields() {
            match step_type {
                StepType::Rank(path_name, target_rank) => {
                    if let Some(id) = self.storage.graph.get_path_id(path_name.as_bytes()) {
                        if self.is_vocab(self.predicate.as_ref(), vg::NODE_PRED) {
                            print!(">");
                            let step = StepPtr::from_one_based(target_rank as usize);
                            self.step = Some(StepInfos(step, target_rank, 3));
                        } else {
                            let path_ref = self.storage.graph.get_path_ref(id).unwrap();
                            let step_handle = path_ref.step_at(path_ref.first_step());
                            let mut step_handle = step_handle.unwrap();
                            let mut node_handle = step_handle.handle();
                            let mut rank = FIRST_RANK;
                            let mut position = FIRST_POS;

                            let steps = self.storage.graph.path_steps(id).expect("Path has steps");
                            // TODO: for -> probably cannot/does not make sense be parallelized
                            for _ in steps.skip(1) {
                                if rank >= target_rank {
                                    break;
                                }
                                step_handle = path_ref.next_step(step_handle.0).unwrap();
                                position += self.storage.graph.node_len(node_handle) as u64;
                                node_handle = step_handle.handle();
                                rank += 1;
                            }
                            self.step = Some(StepInfos(step_handle.0, rank, position));
                        }
                        // println!("First step: {}, {}, {:?}", rank, target_rank, self.get_path_name(id));
                        self.curr_path = Some(id);
                    }
                }
                StepType::Position(path_name, target_pos) => {}
            }
        }
    }

    fn step_no_subject(&mut self) -> Option<EncodedQuad> {
        // print!(",");
        // if self.mode == IterMode::Single && self.sub_mode == SubMode::Step(StepState::NodeReverse) {
        //     println!(
        //         "q||| {:?} {:?} {:?}",
        //         self.subject, self.predicate, self.object
        //     );
        // }
        if let Some(path_id) = self.curr_path {
            if let Some(StepInfos(step, rank, position)) = self.step {
                let path_name = self.get_path_name(path_id).unwrap();
                // print!("576 [ ");
                let node_handle = self
                    .storage
                    .graph
                    .path_handle_at_step(path_id, step)
                    .expect("There should always be a node to every step");
                // println!("] 576");
                let triple = self.step_handle_to_triples(&path_name, node_handle, rank, position);
                // println!("");
                return triple;
            }
        }
        None
    }

    fn steps(&mut self) -> Option<EncodedQuad> {
        if self.subject.is_none() {
            // println!("SF: none self.subject");
            let mut triple = self.step_no_subject();
            while triple.is_none() {
                if let Some(path_id) = self.curr_path {
                    if let Some(StepInfos(step, rank, position)) = self.step {
                        let node_handle = self
                            .storage
                            .graph
                            .path_handle_at_step(path_id, step)
                            .expect("There should always be a node to every step");
                        if let Some(next_step) = self.storage.graph.path_next_step(path_id, step) {
                            // println!("] 586");
                            // print!("589 [ ");
                            let node_length = self.storage.graph.node_len(node_handle) as u64;
                            // println!("] 589");
                            self.step =
                                Some(StepInfos(next_step, rank + 1, position + node_length));
                            self.sub_mode = SubMode::Step(StepState::TypeStep);
                        } else {
                            self.curr_path = self.path_ids.next();
                            if let Some(new_path_id) = self.curr_path {
                                // print!("598 [ ");
                                let next = self
                                    .storage
                                    .graph
                                    .path_first_step(new_path_id)
                                    .expect("Every path should have at least one step");
                                // println!("] 598");
                                self.step = Some(StepInfos(next, FIRST_RANK, FIRST_POS));
                                self.sub_mode = SubMode::Step(StepState::TypeStep);
                            } else {
                                return None;
                            }
                        }
                        triple = self.step_no_subject();
                    } else {
                        panic!("Could not get step info"); // each path should have at least one
                                                           // step?!
                    }
                } else {
                    return None; // Case of having no paths
                }
            }
            return triple;
        } else if let Some(step_type) = self.get_step_iri_fields() {
            // println!("SF: some self.subject");
            // print!("|");
            match step_type {
                StepType::Rank(path_name, _) => {
                    if let Some(path_id) = self.curr_path {
                        if let Some(StepInfos(step, rank, position)) = self.step {
                            if path_name == "HG00673#1#JAHBBZ010000052.1" {
                                // println!("At path: {}, {:?}, {:?}", path_name, self.storage.graph.path_len(path_id), self.get_path_name(path_id));
                                // println!("At path: {:?}, {}, {}", step, rank, position);
                            }
                            // print!("620 [ ");
                            let node_handle = self
                                .storage
                                .graph
                                .path_handle_at_step(path_id, step)
                                .expect("There should always be a node to every step");
                            // println!("] 620");
                            let triple = self.step_handle_to_triples(
                                &path_name,
                                node_handle,
                                rank,
                                position,
                            ); //results.append(&mut triples);
                            if triple.is_none() {
                                self.mode = IterMode::Finished;
                            }
                            return triple;
                        }
                    }
                }
                StepType::Position(path_name, position) => {
                    // println!("POSITION: {}, {}", path_name, position);
                    // print!("638 [ ");
                    if let Some(id) = self.storage.graph.get_path_id(path_name.as_bytes()) {
                        // println!("] 638");
                        // print!("641 [ ");
                        if let Some(step) =
                            self.storage.graph.path_step_at_base(id, position as usize)
                        {
                            // println!("] 641");
                            // print!("645 [ ");
                            let node_handle =
                                self.storage.graph.path_handle_at_step(id, step).unwrap();
                            // println!("] 645");
                            let rank = step.pack() + 1;
                            let triple = self.step_handle_to_triples(
                                &path_name,
                                node_handle,
                                rank,
                                position,
                            ); //results.append(&mut triples);

                            if triple.is_none() {
                                self.mode = IterMode::Finished;
                            }
                            return triple;
                        }
                    }
                }
            }
            //results
        }
        None
    }

    fn get_step_iri_fields(&self) -> Option<StepType> {
        let term = self.subject.as_ref()?;
        if let EncodedTerm::NamedNode { iri_id: _, value } = term {
            let mut parts = value.split("/").collect::<Vec<_>>();
            parts.reverse();
            if parts.len() < 5 || !parts.contains(&"path") {
                // println!("We are quitting early! {:?}", parts);
                return None;
            }
            match parts[1] {
                "step" => {
                    let step_idx = value.rfind("step").expect("Should contain step");
                    let start = self.storage.base.len() + 1 + LEN_OF_PATH_AND_SLASH;
                    let path_text = &value[start..step_idx - 1]; //.replace("/", "#");
                    let path_name = decode(&path_text).ok()?.to_string();
                    // println!("Step: {}\t{}\t{}", step_idx, path_text, path_name);
                    Some(StepType::Rank(path_name, parts[0].parse().ok()?))
                }
                "position" => {
                    let pos_idx = value.rfind("position").expect("Should contain position");
                    let start = self.storage.base.len() + 1 + LEN_OF_PATH_AND_SLASH;
                    let path_text = &value[start..pos_idx - 1]; //.replace("/", "#");
                    let path_name = decode(&path_text).ok()?.to_string();
                    // println!("Pos: {}\t{}\t{}", pos_idx, path_text, path_name);
                    Some(StepType::Position(path_name, parts[0].parse().ok()?))
                }
                _ => None,
            }
        } else {
            None
        }
    }

    fn needs_faldo_triple(&self) -> bool {
        self.predicate.is_none()
            || self.is_vocab(self.predicate.as_ref(), faldo::POSITION_PRED)
            || self.is_vocab(self.predicate.as_ref(), rdf::TYPE)
            || self.is_vocab(self.predicate.as_ref(), faldo::REFERENCE)
    }

    fn step_handle_to_triples(
        &mut self,
        path_name: &str,
        node_handle: Handle,
        rank: u64,
        position: u64,
    ) -> Option<EncodedQuad> {
        let step_iri = self.step_to_namednode(path_name, rank).unwrap();
        let node_len = self.storage.graph.node_len(node_handle) as u64;
        let position_literal = EncodedTerm::IntegerLiteral((position as i64).into());
        // println!("SH");
        // print!(".");

        if self.subject.is_none() || step_iri == self.subject.as_ref().unwrap().clone() {
            if let SubMode::Step(st) = self.sub_mode {
                match st {
                    StepState::TypeStep => {
                        self.sub_mode = SubMode::Step(StepState::TypeRegion);
                        self.get_step_type_step(step_iri).or_else(|| {
                            self.step_handle_to_triples(path_name, node_handle, rank, position)
                        })
                    }
                    StepState::TypeRegion => {
                        self.sub_mode = SubMode::Step(StepState::Node);
                        self.get_step_type_region(step_iri).or_else(|| {
                            self.step_handle_to_triples(path_name, node_handle, rank, position)
                        })
                    }
                    StepState::Node => {
                        self.sub_mode = SubMode::Step(StepState::NodeReverse);
                        self.get_step_node(step_iri, node_handle, false)
                            .or_else(|| {
                                self.step_handle_to_triples(path_name, node_handle, rank, position)
                            })
                    }
                    StepState::NodeReverse => {
                        self.sub_mode = SubMode::Step(StepState::Rank);
                        self.get_step_node(step_iri, node_handle, true).or_else(|| {
                            self.step_handle_to_triples(path_name, node_handle, rank, position)
                        })
                    }
                    StepState::Rank => {
                        self.sub_mode = SubMode::Step(StepState::Position);
                        self.get_step_rank(step_iri, rank).or_else(|| {
                            self.step_handle_to_triples(path_name, node_handle, rank, position)
                        })
                    }
                    StepState::Position => {
                        self.sub_mode = SubMode::Step(StepState::Path);
                        self.get_step_position(step_iri, position_literal)
                            .or_else(|| {
                                self.step_handle_to_triples(path_name, node_handle, rank, position)
                            })
                    }
                    StepState::Path => {
                        self.sub_mode = SubMode::Step(StepState::Begin);
                        let path_iri = self.path_to_namednode(path_name).unwrap();
                        self.get_step_path(step_iri, path_iri).or_else(|| {
                            self.step_handle_to_triples(path_name, node_handle, rank, position)
                        })
                    }
                    StepState::Begin => {
                        self.sub_mode = SubMode::Step(StepState::End);
                        self.get_step_terminal(
                            step_iri,
                            position_literal,
                            node_len,
                            false,
                            path_name,
                            position,
                        )
                        .or_else(|| {
                            self.step_handle_to_triples(path_name, node_handle, rank, position)
                        })
                    }
                    StepState::End => {
                        if self.subject.is_none() && self.needs_faldo_triple() {
                            self.sub_mode =
                                SubMode::Step(StepState::FaldoBegin(FaldoState::Positon));
                        } else {
                            self.sub_mode = SubMode::Step(StepState::Finished);
                        }
                        self.get_step_terminal(
                            step_iri,
                            position_literal,
                            node_len,
                            true,
                            path_name,
                            position,
                        )
                        .or_else(|| {
                            self.step_handle_to_triples(path_name, node_handle, rank, position)
                        })
                    }
                    StepState::FaldoBegin(_) => {
                        let subject = self
                            .get_faldo_border_namednode(position, path_name)
                            .unwrap();
                        let path_iri = self.path_to_namednode(path_name).unwrap();
                        self.faldo_for_step(position, path_iri, &subject)
                            .or_else(|| {
                                self.sub_mode =
                                    SubMode::Step(StepState::FaldoEnd(FaldoState::Positon));
                                self.step_handle_to_triples(path_name, node_handle, rank, position)
                            })
                    }
                    StepState::FaldoEnd(_) => {
                        let subject = self
                            .get_faldo_border_namednode(position + node_len, path_name)
                            .unwrap();
                        let path_iri = self.path_to_namednode(path_name).unwrap();
                        self.faldo_for_step(position + node_len, path_iri, &subject)
                            .or_else(|| {
                                self.sub_mode = SubMode::Step(StepState::Finished);
                                None
                            })
                    }
                    StepState::Finished => None,
                }
            } else {
                panic!("When getting steps, we should always be in step mode");
            }
        } else {
            None
        }
    }

    fn get_step_type_step(&self, step_iri: EncodedTerm) -> Option<EncodedQuad> {
        if self.is_vocab(self.predicate.as_ref(), rdf::TYPE) || self.predicate.is_none() {
            if self.object.is_none() || self.is_vocab(self.object.as_ref(), vg::STEP) {
                // println!("SH: none/type self.predicate");
                return Some(EncodedQuad::new(
                    step_iri.clone(),
                    rdf::TYPE.into(),
                    vg::STEP.into(),
                    self.graph_name.clone(),
                ));
            }
        }
        None
    }

    fn get_step_type_region(&self, step_iri: EncodedTerm) -> Option<EncodedQuad> {
        if self.is_vocab(self.predicate.as_ref(), rdf::TYPE) || self.predicate.is_none() {
            if self.object.is_none() || self.is_vocab(self.object.as_ref(), faldo::REGION) {
                // println!("SH: region self.object");
                return Some(EncodedQuad::new(
                    step_iri.clone(),
                    rdf::TYPE.into(),
                    faldo::REGION.into(),
                    self.graph_name.clone(),
                ));
            }
        }
        None
    }

    fn get_step_node(
        &self,
        step_iri: EncodedTerm,
        node_handle: Handle,
        is_reverse: bool,
    ) -> Option<EncodedQuad> {
        let pred = match is_reverse {
            false => vg::NODE_PRED,
            true => vg::REVERSE_OF_NODE,
        };
        let node_handle_is_reverse = is_reverse == node_handle.is_reverse();
        let node_iri = self.handle_to_namednode(node_handle).unwrap();
        if (self.is_vocab(self.predicate.as_ref(), pred)
            || self.predicate.is_none() && node_handle_is_reverse)
            && (self.object.is_none() || node_iri == self.object.as_ref().unwrap().clone())
        {
            // println!("SH: node self.object");
            return Some(EncodedQuad::new(
                step_iri.clone(),
                pred.into(),
                node_iri.clone(),
                self.graph_name.clone(),
            ));
        }
        None
    }

    fn get_step_rank(&self, step_iri: EncodedTerm, rank: u64) -> Option<EncodedQuad> {
        if self.is_vocab(self.predicate.as_ref(), vg::RANK) || self.predicate.is_none() {
            let rank_literal = EncodedTerm::IntegerLiteral((rank as i64).into());
            if self.object.is_none() || self.object.as_ref().unwrap().clone() == rank_literal {
                // println!("SH: rank self.predicate");
                return Some(EncodedQuad::new(
                    step_iri.clone(),
                    vg::RANK.into(),
                    rank_literal,
                    self.graph_name.clone(),
                ));
            }
        }
        None
    }

    fn get_step_position(
        &self,
        step_iri: EncodedTerm,
        position_literal: EncodedTerm,
    ) -> Option<EncodedQuad> {
        if self.is_vocab(self.predicate.as_ref(), vg::POSITION) || self.predicate.is_none() {
            if self.object.is_none() || self.object.as_ref().unwrap().clone() == position_literal {
                // println!("SH: position self.predicate");
                return Some(EncodedQuad::new(
                    step_iri.clone(),
                    vg::POSITION.into(),
                    position_literal.clone(),
                    self.graph_name.clone(),
                ));
            }
        }
        None
    }

    fn get_step_path(&self, step_iri: EncodedTerm, path_iri: EncodedTerm) -> Option<EncodedQuad> {
        if self.is_vocab(self.predicate.as_ref(), vg::PATH_PRED) || self.predicate.is_none() {
            if self.object.is_none() || path_iri == self.object.as_ref().unwrap().clone() {
                // println!("SH: path self.predicate");
                return Some(EncodedQuad::new(
                    step_iri.clone(),
                    vg::PATH_PRED.into(),
                    path_iri.clone(),
                    self.graph_name.clone(),
                ));
            }
        }
        None
    }

    fn get_step_terminal(
        &self,
        step_iri: EncodedTerm,
        position_literal: EncodedTerm,
        node_len: u64,
        is_end: bool,
        path_name: &str,
        position: u64,
    ) -> Option<EncodedQuad> {
        let pred = match is_end {
            false => faldo::BEGIN,
            true => faldo::END,
        };
        let pos = match is_end {
            false => position,
            true => position + node_len,
        };
        if self.predicate.is_none() || self.is_vocab(self.predicate.as_ref(), pred) {
            if self.object.is_none() || self.object.as_ref().unwrap().clone() == position_literal {
                // println!("SH: begin self.predicate");
                return Some(EncodedQuad::new(
                    step_iri.clone(),
                    pred.into(),
                    self.get_faldo_border_namednode(pos, path_name).unwrap(), // FIX
                    self.graph_name.clone(),
                ));
            }
        }
        None
    }

    fn get_faldo_border_namednode(&self, position: u64, path_name: &str) -> Option<EncodedTerm> {
        // let path_name = path_name.replace("#", "/");
        let text = format!(
            "{}/path/{}/position/{}",
            self.storage.base, path_name, position
        );
        Some(EncodedTerm::NamedNode {
            iri_id: StrHash::new(""),
            value: text,
        })
    }

    fn faldo_for_step(
        &mut self,
        position: u64,
        path_iri: EncodedTerm,
        subject: &EncodedTerm,
    ) -> Option<EncodedQuad> {
        if let SubMode::Step(faldo) = self.sub_mode {
            let (triple, new_fst) = match faldo {
                StepState::FaldoBegin(fst) | StepState::FaldoEnd(fst) => match fst {
                    FaldoState::Positon => {
                        (self.get_faldo_pos(subject, position), FaldoState::Exact)
                    }
                    FaldoState::Exact => (self.get_faldo_exact(subject), FaldoState::Type),
                    FaldoState::Type => (self.get_faldo_type(subject), FaldoState::Reference),
                    FaldoState::Reference => (
                        self.get_faldo_reference(subject, path_iri.clone()),
                        FaldoState::Finished,
                    ),
                    FaldoState::Finished => (None, FaldoState::Finished),
                },
                _ => panic!("Should never calculate faldo for non-faldo"),
            };
            match faldo {
                StepState::FaldoBegin(_) => {
                    self.sub_mode = SubMode::Step(StepState::FaldoBegin(new_fst))
                }
                StepState::FaldoEnd(_) => {
                    self.sub_mode = SubMode::Step(StepState::FaldoEnd(new_fst))
                }
                _ => (),
            };
            triple.or_else(|| {
                if new_fst != FaldoState::Finished {
                    self.faldo_for_step(position, path_iri, subject)
                } else {
                    None
                }
            })
        } else {
            panic!("Should never calculate faldo for non-step");
        }
    }

    fn get_faldo_pos(&self, subject: &EncodedTerm, position: u64) -> Option<EncodedQuad> {
        let ep = EncodedTerm::IntegerLiteral((position as i64).into());
        if (self.predicate.is_none()
            || self.is_vocab(self.predicate.as_ref(), faldo::POSITION_PRED))
            && (self.object.is_none() || self.object.as_ref() == Some(&ep))
        {
            // println!("FS: position");
            Some(EncodedQuad::new(
                subject.clone(),
                faldo::POSITION_PRED.into(),
                ep,
                self.graph_name.clone(),
            ))
        } else {
            None
        }
    }

    fn get_faldo_exact(&self, subject: &EncodedTerm) -> Option<EncodedQuad> {
        if (self.predicate.is_none() || self.is_vocab(self.predicate.as_ref(), rdf::TYPE))
            && (self.object.is_none() || self.is_vocab(self.object.as_ref(), faldo::EXACT_POSITION))
        {
            // println!("FS: position");
            Some(EncodedQuad::new(
                subject.clone(),
                rdf::TYPE.into(),
                faldo::EXACT_POSITION.into(),
                self.graph_name.clone(),
            ))
        } else {
            None
        }
    }

    fn get_faldo_type(&self, subject: &EncodedTerm) -> Option<EncodedQuad> {
        if (self.predicate.is_none() || self.is_vocab(self.predicate.as_ref(), rdf::TYPE))
            && (self.object.is_none() || self.is_vocab(self.object.as_ref(), faldo::POSITION))
        {
            Some(EncodedQuad::new(
                subject.clone(),
                rdf::TYPE.into(),
                faldo::POSITION.into(),
                self.graph_name.clone(),
            ))
        } else {
            None
        }
    }

    fn get_faldo_reference(
        &self,
        subject: &EncodedTerm,
        path_iri: EncodedTerm,
    ) -> Option<EncodedQuad> {
        if (self.predicate.is_none() || self.is_vocab(self.predicate.as_ref(), faldo::REFERENCE))
            && (self.object.is_none() || self.object.as_ref() == Some(&path_iri))
        {
            Some(EncodedQuad::new(
                subject.clone(),
                faldo::REFERENCE.into(),
                path_iri,
                self.graph_name.clone(),
            ))
        } else {
            None
        }
    }

    fn handle_to_triples(&self, handle: Handle) -> Option<EncodedQuad> {
        if self.is_vocab(self.predicate.as_ref(), rdf::VALUE) || self.predicate.is_none() {
            let seq_bytes = self.storage.graph.sequence_vec(handle);
            let seq = str::from_utf8(&seq_bytes).expect("Node contains sequence");
            let seq_value = Literal::new_simple_literal(seq);
            if self.object.is_none()
                || self.decode_term(self.object.as_ref().unwrap()).unwrap()
                    == Term::Literal(seq_value.clone())
            {
                Some(EncodedQuad::new(
                    self.handle_to_namednode(handle).unwrap(),
                    rdf::VALUE.into(),
                    seq_value.as_ref().into(),
                    self.graph_name.clone(),
                ))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Knows that we should generate an edge triple for handle,
    /// self.edges contains edges
    fn handle_to_edge_triples(
        &mut self,
        handle: Handle,
        is_directional: bool,
    ) -> Option<EncodedQuad> {
        if self.predicate.is_none() || self.is_node_related() {
            let neighbor = self.edges.next()?;
            if self.object.is_none()
                || self
                    .get_node_id(self.object.as_ref().unwrap())
                    .expect("Object has node id")
                    == neighbor.unpack_number()
            {
                Some(self.handles_to_edge_triple(handle, neighbor, is_directional))
            } else {
                self.handle_to_edge_triples(handle, is_directional)
            }
        } else {
            None
        }
    }

    fn handles_to_edge_triple(&self, a: Handle, b: Handle, is_directional: bool) -> EncodedQuad {
        if !is_directional {
            EncodedQuad::new(
                self.handle_to_namednode(a).unwrap(),
                vg::LINKS.into(),
                self.handle_to_namednode(b).unwrap(),
                self.graph_name.clone(),
            )
        } else {
            EncodedQuad::new(
                self.handle_to_namednode(a).unwrap(),
                vg::LINKS_FORWARD_TO_FORWARD.into(), // TODO: Other directions
                self.handle_to_namednode(b).unwrap(),
                self.graph_name.clone(),
            )
        }
    }

    fn generate_edge_triples<'b>(
        &'b self,
        subject: Handle,
        object: Handle,
    ) -> impl Iterator<Item = EncodedQuad> + 'b {
        gen!({
            let node_is_reverse = subject.is_reverse();
            let other_is_reverse = object.is_reverse();
            if (self.predicate.is_none()
                || self.is_vocab(self.predicate.as_ref(), vg::LINKS_FORWARD_TO_FORWARD))
                && !node_is_reverse
                && !other_is_reverse
            {
                yield_!(EncodedQuad::new(
                    self.handle_to_namednode(subject).expect("Subject is fine"),
                    vg::LINKS_FORWARD_TO_FORWARD.into(),
                    self.handle_to_namednode(object).expect("Object is fine"),
                    self.graph_name.clone(),
                ));
            }
            if (self.predicate.is_none()
                || self.is_vocab(self.predicate.as_ref(), vg::LINKS_FORWARD_TO_REVERSE))
                && !node_is_reverse
                && other_is_reverse
            {
                yield_!(EncodedQuad::new(
                    self.handle_to_namednode(subject).expect("Subject is fine"),
                    vg::LINKS_FORWARD_TO_REVERSE.into(),
                    self.handle_to_namednode(object).expect("Object is fine"),
                    self.graph_name.clone(),
                ));
            }
            if (self.predicate.is_none()
                || self.is_vocab(self.predicate.as_ref(), vg::LINKS_REVERSE_TO_FORWARD))
                && node_is_reverse
                && !other_is_reverse
            {
                yield_!(EncodedQuad::new(
                    self.handle_to_namednode(subject).expect("Subject is fine"),
                    vg::LINKS_REVERSE_TO_FORWARD.into(),
                    self.handle_to_namednode(object).expect("Object is fine"),
                    self.graph_name.clone(),
                ));
            }
            if (self.predicate.is_none()
                || self.is_vocab(self.predicate.as_ref(), vg::LINKS_REVERSE_TO_REVERSE))
                && node_is_reverse
                && other_is_reverse
            {
                yield_!(EncodedQuad::new(
                    self.handle_to_namednode(subject).expect("Subject is fine"),
                    vg::LINKS_REVERSE_TO_REVERSE.into(),
                    self.handle_to_namednode(object).expect("Object is fine"),
                    self.graph_name.clone(),
                ));
            }
            if self.predicate.is_none() || self.is_vocab(self.predicate.as_ref(), vg::LINKS) {
                yield_!(EncodedQuad::new(
                    self.handle_to_namednode(subject).expect("Subject is fine"),
                    vg::LINKS.into(),
                    self.handle_to_namednode(object).expect("Object is fine"),
                    self.graph_name.clone(),
                ));
            }
        })
        .into_iter()
    }

    fn handle_to_namednode(&self, handle: Handle) -> Option<EncodedTerm> {
        let id = handle.unpack_number();
        let text = format!("{}/node/{}", self.storage.base, id);
        Some(EncodedTerm::NamedNode {
            iri_id: StrHash::new(""),
            value: text,
        })
    }

    fn step_to_namednode(&self, path_name: &str, rank: u64) -> Option<EncodedTerm> {
        // println!("STEP_TO_NAMEDNODE: {} - {:?}", path_name, rank);
        // let path_name = path_name.replace("#", "/");
        let text = format!("{}/path/{}/step/{}", self.storage.base, path_name, rank);
        Some(EncodedTerm::NamedNode {
            iri_id: StrHash::new(""),
            value: text,
        })
    }

    fn path_to_namednode(&self, path_name: &str) -> Option<EncodedTerm> {
        // println!("PATH_TO_NAMEDNODE: {}", path_name);
        // let path_name = path_name.replace("#", "/");
        let text = format!("{}/path/{}", self.storage.base, path_name);
        Some(EncodedTerm::NamedNode {
            iri_id: StrHash::new(""),
            value: text,
        })
    }

    fn get_path_name(&self, path_id: PathId) -> Option<String> {
        if let Some(path_name_iter) = self.storage.graph.get_path_name(path_id) {
            let path_name: Vec<u8> = path_name_iter.collect();
            let path_name = std::str::from_utf8(&path_name).ok()?;
            Some(path_name.to_owned())
        } else {
            None
        }
    }

    fn is_node_related(&self) -> bool {
        let predicates = [
            vg::LINKS,
            vg::LINKS_FORWARD_TO_FORWARD,
            vg::LINKS_FORWARD_TO_REVERSE,
            vg::LINKS_REVERSE_TO_FORWARD,
            vg::LINKS_REVERSE_TO_REVERSE,
        ];
        if self.predicate.is_none() {
            return false;
        }
        predicates
            .into_iter()
            .map(|x| self.is_vocab(self.predicate.as_ref(), x))
            .reduce(|acc, x| acc || x)
            .unwrap()
    }

    fn is_step_associated_type(&self) -> bool {
        let types = [
            faldo::REGION,
            faldo::EXACT_POSITION,
            faldo::POSITION,
            vg::STEP,
        ];
        if self.object.is_none() {
            return false;
        }
        types
            .into_iter()
            .map(|x| self.is_vocab(self.object.as_ref(), x))
            .reduce(|acc, x| acc || x)
            .unwrap()
    }

    fn is_step_associated(&self) -> bool {
        let predicates = [
            vg::RANK,
            vg::POSITION,
            vg::PATH_PRED,
            vg::NODE_PRED,
            vg::REVERSE_OF_NODE,
            faldo::BEGIN,
            faldo::END,
            faldo::REFERENCE,
            faldo::POSITION_PRED,
        ];
        if self.predicate.is_none() {
            return false;
        }
        predicates
            .into_iter()
            .map(|x| self.is_vocab(self.predicate.as_ref(), x))
            .reduce(|acc, x| acc || x)
            .unwrap()
    }

    fn is_vocab(&self, term: Option<&EncodedTerm>, vocab: NamedNodeRef) -> bool {
        if term.is_none() {
            return false;
        }
        let term = term.unwrap();
        if !term.is_named_node() {
            return false;
        }
        let named_node = term.get_named_node_value().expect("Is named node");
        named_node == vocab.as_str()
    }

    fn is_node_iri_in_graph(&self, term: &EncodedTerm) -> bool {
        match self.get_node_id(term) {
            Some(id) => self.storage.graph.has_node(id),
            None => false,
        }
    }

    fn get_node_id(&self, term: &EncodedTerm) -> Option<u64> {
        match term.is_named_node() {
            true => {
                let text = term
                    .get_named_node_value()
                    .expect("Encoded NamedNode has to have value")
                    .clone();

                // Remove trailing '>'
                // text.pop();

                let mut parts_iter = text.rsplit("/");
                let last = parts_iter.next();
                let pre_last = parts_iter.next();
                match last.is_some()
                    && pre_last.is_some()
                    && pre_last.expect("Option is some") == "node"
                {
                    true => last.expect("Option is some").parse::<u64>().ok(),
                    false => None,
                }
            }
            false => None,
        }
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn get_str(&self, _key: &StrHash) -> Result<Option<String>, StorageError> {
        Ok(None)
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn contains_str(&self, _key: &StrHash) -> Result<bool, StorageError> {
        Ok(true)
    }
}

impl StrLookup for GraphIter {
    fn get_str(&self, key: &StrHash) -> Result<Option<String>, StorageError> {
        self.get_str(key)
    }

    fn contains_str(&self, key: &StrHash) -> Result<bool, StorageError> {
        self.contains_str(key)
    }
}

enum StepType {
    Rank(String, u64),
    Position(String, u64),
}

enum SubjectType {
    PathIri,
    StepBorderIri,
    NodeIri,
    StepIri,
}

#[cfg(test)]
mod tests {
    use std::{path::Path, str::FromStr};

    use crate::storage::small_string::SmallString;

    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    const BASE: &'static str = "https://example.org";

    fn _get_generator(gfa: &str) -> StorageGenerator {
        let storage = Storage::from_str(gfa).unwrap();
        StorageGenerator::new(storage)
    }

    fn get_odgi_test_file_generator(file_name: &str) -> StorageGenerator {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(file_name);
        let storage = Storage::open(&path).unwrap();
        StorageGenerator::new(storage)
    }

    fn print_quad(quad: &EncodedQuad) {
        let sub = match &quad.subject {
            EncodedTerm::NamedNode { iri_id: _, value } => value.clone(),
            _ => "NOT NAMED".to_owned(),
        };
        let pre = match &quad.predicate {
            EncodedTerm::NamedNode { iri_id: _, value } => value.clone(),
            _ => "NOT NAMED".to_owned(),
        };
        let obj = match &quad.object {
            EncodedTerm::NamedNode { iri_id: _, value } => value.clone(),
            EncodedTerm::SmallStringLiteral(value) => format!("\"{}\"", value).to_string(),
            EncodedTerm::IntegerLiteral(value) => value.to_string(),
            _ => "NOT NAMED".to_owned(),
        };
        println!("{}\t{}\t{} .", sub, pre, obj);
    }

    fn get_node(id: i64) -> EncodedTerm {
        let text = format!("{}/node/{}", BASE, id);
        EncodedTerm::NamedNode {
            iri_id: StrHash::new(""),
            value: text,
        }
    }

    fn get_step(path: &str, id: i64) -> EncodedTerm {
        // let path = path.replace("#", "/");
        let text = format!("{}/path/{}/step/{}", BASE, path, id);
        EncodedTerm::NamedNode {
            iri_id: StrHash::new(""),
            value: text,
        }
    }

    fn get_position(path: &str, id: i64) -> EncodedTerm {
        // let path = path.replace("#", "/");
        let text = format!("{}/path/{}/position/{}", BASE, path, id);
        EncodedTerm::NamedNode {
            iri_id: StrHash::new(""),
            value: text,
        }
    }

    fn get_path(path: &str) -> EncodedTerm {
        // let path = path.replace("#", "/");
        let text = format!("{}/path/{}", BASE, path);
        EncodedTerm::NamedNode {
            iri_id: StrHash::new(""),
            value: text,
        }
    }

    fn count_subjects(subject: &EncodedTerm, triples: &Vec<EncodedQuad>) -> usize {
        let mut count = 0;
        for triple in triples {
            if &triple.subject == subject {
                count += 1;
            }
        }
        count
    }

    #[test]
    fn test_single_node() {
        let gen = get_odgi_test_file_generator("t_red.gfa");
        let node_triple: Vec<_> = gen
            .quads_for_pattern(None, None, None, &EncodedTerm::DefaultGraph)
            .flat_map(|x| x)
            .collect();
        let node_id_quad = EncodedQuad::new(
            get_node(1),
            rdf::TYPE.into(),
            vg::NODE.into(),
            EncodedTerm::DefaultGraph,
        );
        let sequence_quad = EncodedQuad::new(
            get_node(1),
            rdf::VALUE.into(),
            EncodedTerm::SmallStringLiteral(SmallString::from_str("CAAATAAG").unwrap()),
            EncodedTerm::DefaultGraph,
        );
        assert_eq!(node_triple.len(), 2);
        assert!(node_triple.contains(&node_id_quad));
        assert!(node_triple.contains(&sequence_quad));
    }

    #[test]
    fn test_single_node_type_spo() {
        let gen = get_odgi_test_file_generator("t_red.gfa");
        let node_1 = get_node(1);
        let node_triple: Vec<_> = gen
            .quads_for_pattern(
                Some(&node_1),
                Some(&rdf::TYPE.into()),
                Some(&vg::NODE.into()),
                &EncodedTerm::DefaultGraph,
            )
            .flat_map(|x| x)
            .collect();
        let node_id_quad = EncodedQuad::new(
            get_node(1),
            rdf::TYPE.into(),
            vg::NODE.into(),
            EncodedTerm::DefaultGraph,
        );
        for tripe in &node_triple {
            print_quad(tripe);
        }
        assert_eq!(node_triple.len(), 1);
        assert!(node_triple.contains(&node_id_quad));
    }

    #[test]
    fn test_single_node_type_s() {
        let gen = get_odgi_test_file_generator("t_red.gfa");
        let node_triple: Vec<_> = gen
            .quads_for_pattern(Some(&get_node(1)), None, None, &EncodedTerm::DefaultGraph)
            .flat_map(|x| x)
            .collect();
        let node_id_quad = EncodedQuad::new(
            get_node(1),
            rdf::TYPE.into(),
            vg::NODE.into(),
            EncodedTerm::DefaultGraph,
        );
        let sequence_quad = EncodedQuad::new(
            get_node(1),
            rdf::VALUE.into(),
            EncodedTerm::SmallStringLiteral(SmallString::from_str("CAAATAAG").unwrap()),
            EncodedTerm::DefaultGraph,
        );
        for tripe in &node_triple {
            print_quad(tripe);
        }
        assert_eq!(node_triple.len(), 2);
        assert!(node_triple.contains(&node_id_quad));
        assert!(node_triple.contains(&sequence_quad));
    }

    #[ignore]
    #[test]
    fn test_single_node_type_p() {
        let gen = get_odgi_test_file_generator("t_red.gfa");
        let node_triple: Vec<_> = gen
            .quads_for_pattern(
                None,
                Some(&rdf::TYPE.into()),
                None,
                &EncodedTerm::DefaultGraph,
            )
            .flat_map(|x| x)
            .collect();
        let node_id_quad = EncodedQuad::new(
            get_node(1),
            rdf::TYPE.into(),
            vg::NODE.into(),
            EncodedTerm::DefaultGraph,
        );
        for tripe in &node_triple {
            print_quad(tripe);
        }
        assert_eq!(node_triple.len(), 1);
        assert!(node_triple.contains(&node_id_quad));
    }

    #[ignore]
    #[test]
    fn test_single_node_type_o() {
        let gen = get_odgi_test_file_generator("t_red.gfa");
        let node_triple: Vec<_> = gen
            .quads_for_pattern(
                None,
                None,
                Some(&vg::NODE.into()),
                &EncodedTerm::DefaultGraph,
            )
            .flat_map(|x| x)
            .collect();
        let node_id_quad = EncodedQuad::new(
            get_node(1),
            rdf::TYPE.into(),
            vg::NODE.into(),
            EncodedTerm::DefaultGraph,
        );
        for tripe in &node_triple {
            print_quad(tripe);
        }
        assert_eq!(node_triple.len(), 1);
        assert!(node_triple.contains(&node_id_quad));
    }

    #[test]
    fn test_double_node() {
        // Reminder: fails with "old" version of rs-handlegraph (use git-master)
        let gen = get_odgi_test_file_generator("t_double.gfa");
        let node_triple: Vec<_> = gen
            .quads_for_pattern(None, None, None, &EncodedTerm::DefaultGraph)
            .flat_map(|x| x)
            .collect();
        let links_quad = EncodedQuad::new(
            get_node(1),
            vg::LINKS.into(),
            get_node(2),
            EncodedTerm::DefaultGraph,
        );
        let links_f2f_quad = EncodedQuad::new(
            get_node(1),
            vg::LINKS_FORWARD_TO_FORWARD.into(),
            get_node(2),
            EncodedTerm::DefaultGraph,
        );
        for tripe in &node_triple {
            print_quad(tripe);
        }
        assert_eq!(node_triple.len(), 39);
        assert!(node_triple.contains(&links_quad));
        assert!(node_triple.contains(&links_f2f_quad));
    }

    #[test]
    // TODO: Fix position numbers e.g. having pos/1 + pos/9 and pos/9 + pos/10
    fn test_step() {
        let gen = get_odgi_test_file_generator("t_step.gfa");
        let step_triples: Vec<_> = gen
            .quads_for_pattern(None, None, None, &EncodedTerm::DefaultGraph)
            .flat_map(|x| x)
            .collect();
        for triple in &step_triples {
            print_quad(triple);
        }
        let count_step1 = count_subjects(&get_step("x/a", 1), &step_triples);
        let count_step2 = count_subjects(&get_step("x/a", 2), &step_triples);
        let count_pos1 = count_subjects(&get_position("x/a", 1), &step_triples);
        let count_pos9 = count_subjects(&get_position("x/a", 9), &step_triples);
        let count_pos10 = count_subjects(&get_position("x/a", 10), &step_triples);
        assert_eq!(count_step1, 8, "Number of step 1 triples");
        assert_eq!(count_step2, 8, "Number of step 2 triples");
        assert_eq!(count_pos1, 4, "Number of pos 1 triples");
        assert_eq!(count_pos9, 8, "Number of pos 9 triples");
        assert_eq!(count_pos10, 4, "Number of pos 10 triples");
    }

    #[test]
    fn test_step_s() {
        let gen = get_odgi_test_file_generator("t_step.gfa");
        let step_triples: Vec<_> = gen
            .quads_for_pattern(
                Some(&get_step("x/a", 1)),
                None,
                None,
                &EncodedTerm::DefaultGraph,
            )
            .flat_map(|x| x)
            .collect();
        for triple in &step_triples {
            print_quad(triple);
        }
        assert_eq!(step_triples.len(), 8, "Number of step 1 triples");
    }

    #[ignore]
    #[test]
    fn test_step_p() {
        let gen = get_odgi_test_file_generator("t_step.gfa");
        let step_triples: Vec<_> = gen
            .quads_for_pattern(
                None,
                Some(&rdf::TYPE.into()),
                None,
                &EncodedTerm::DefaultGraph,
            )
            .flat_map(|x| x)
            .collect();
        for triple in &step_triples {
            print_quad(triple);
        }
        assert_eq!(step_triples.len(), 12, "Number of type triples");
    }

    #[ignore]
    #[test]
    fn test_step_o() {
        let gen = get_odgi_test_file_generator("t_step.gfa");
        let step_triples: Vec<_> = gen
            .quads_for_pattern(None, None, Some(&get_node(1)), &EncodedTerm::DefaultGraph)
            .flat_map(|x| x)
            .collect();
        for triple in &step_triples {
            print_quad(triple);
        }
        assert_eq!(step_triples.len(), 1, "Number of type triples");
    }

    #[test]
    fn test_step_node() {
        let gen = get_odgi_test_file_generator("t.gfa");
        let step_triples: Vec<_> = gen
            .quads_for_pattern(
                None,
                Some(&vg::NODE_PRED.into()),
                None,
                &EncodedTerm::DefaultGraph,
            )
            .flat_map(|x| x)
            .collect();
        for triple in &step_triples {
            print_quad(triple);
        }
        let quad = EncodedQuad::new(
            get_step("x/a", 6),
            vg::NODE_PRED.into(),
            get_node(9),
            EncodedTerm::DefaultGraph,
        );
        assert_eq!(step_triples.len(), 10, "Number of node_pred triples");
        assert!(step_triples.contains(&quad));
    }

    #[ignore]
    #[test]
    fn test_paths() {
        let gen = get_odgi_test_file_generator("t.gfa");
        let generic_triples: Vec<_> = gen
            .quads_for_pattern(None, None, None, &EncodedTerm::DefaultGraph)
            .flat_map(|x| x)
            .collect();
        let specific_triples: Vec<_> = gen
            .quads_for_pattern(
                Some(&get_path("x/a")),
                Some(&rdf::TYPE.into()),
                Some(&vg::PATH.into()),
                &EncodedTerm::DefaultGraph,
            )
            .flat_map(|x| x)
            .collect();
        for triple in &generic_triples {
            print_quad(triple)
        }
        let quad = EncodedQuad::new(
            get_path("x/a"),
            rdf::TYPE.into(),
            vg::PATH.into(),
            EncodedTerm::DefaultGraph,
        );
        assert_eq!(generic_triples, specific_triples);
        assert_eq!(generic_triples.len(), 1);
        assert!(generic_triples.contains(&quad));
    }

    #[ignore]
    #[test]
    fn test_full() {
        let gen = get_odgi_test_file_generator("t.gfa");
        let node_triple = gen.quads_for_pattern(None, None, None, &EncodedTerm::DefaultGraph);
        for tripe in node_triple {
            print_quad(tripe.as_ref().unwrap());
        }
        assert_eq!(1, 2);
    }
}
