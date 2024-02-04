#![allow(clippy::same_name_method)]
use super::numeric_encoder::{StrHash, StrLookup};
use super::{ChainedDecodingQuadIterator, Storage};
use crate::model::vocab::rdf;
use crate::model::{NamedNodeRef, Term};
use crate::storage::binary_encoder::QuadEncoding;
pub use crate::storage::error::{CorruptionError, LoaderError, SerializerError, StorageError};
use crate::storage::numeric_encoder::Decoder;
#[cfg(not(target_family = "wasm"))]
use crate::storage::numeric_encoder::{EncodedQuad, EncodedTerm};
use crate::storage::vg_vocab::{faldo, vg};
use crate::storage::DecodingQuadIterator;
use genawaiter::{
    rc::{gen, Gen},
    yield_,
};
use gfa::gfa::Orientation;
use handlegraph::handle::{Direction, Handle};
use handlegraph::packed::PackedElement;
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
use std::future::Future;
use std::rc::Rc;
use std::{str, iter};
use urlencoding::{decode, encode};

const LEN_OF_PATH_AND_SLASH: usize = 5;
const URL_HASH: &str = "%23";

pub struct StorageGenerator {
    storage: Rc<Storage>,
}

impl StorageGenerator {
    pub fn new(storage: Storage) -> Self {
        Self { storage: Rc::new(storage) }
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

struct GraphIter {
    storage: Rc<Storage>,
    subject: Option<EncodedTerm>,
    predicate: Option<EncodedTerm>,
    object: Option<EncodedTerm>,
    graph_name: EncodedTerm,
}

impl GraphIter {
    pub fn new(storage: Rc<Storage>, subject: Option<&EncodedTerm>, predicate: Option<&EncodedTerm>, object: Option<&EncodedTerm>, graph_name: &EncodedTerm) -> Self {
        Self {
            storage,
            subject: subject.map(|s| s.to_owned()),
            predicate: predicate.map(|p| p.to_owned()),
            object: object.map(|o| o.to_owned()),
            graph_name: graph_name.to_owned(),
        }
    }

    fn quads_for_pattern<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = EncodedQuad> + 'a> {
        //println!("C:\t{}\t{}\t{} .", self.term_to_text(subject), self.term_to_text(predicate), self.term_to_text(self.object));

        // There should be no blank nodes in the data
        if self.subject.is_some_and(|s| s.is_blank_node()) || self.object.is_some_and(|o| o.is_blank_node()) {
            println!("OF: blanks");
            // yield_!(None);
        }

        if self.is_vocab(self.predicate.as_ref(), rdf::TYPE) && self.object.is_some() {
            return Box::new(self.type_triples());
        } else if self.is_node_related() {
            // println!("OF: nodes");
            let terms = self.nodes(self.subject.as_ref());
            return Box::new(terms);
        } else if self.is_step_associated() {
            // println!("OF: steps");
            let terms = self.steps();
            return Box::new(terms);
        } else if self.is_vocab(self.predicate.as_ref(), rdfs::LABEL) {
            // println!("OF: rdfs::label");
            let terms = self.paths();
            return Box::new(terms);
        } else if self.subject.is_none() && self.predicate.is_none() && self.object.is_none() {
            // println!("OF: triple none");
            let mut terms = self.nodes(self.subject.as_ref());
            let terms_paths = self.paths();
            let terms_steps = self.steps();
            return Box::new(terms.chain(terms_paths).chain(terms_steps));
        } else if self.subject.is_some() {
            // println!("OF: self.subject some");
            let terms: Box<dyn Iterator<Item = EncodedQuad>> = match self.get_term_type(self.subject.as_ref().unwrap()) {
                Some(SubjectType::NodeIri) => {
                    let mut terms =
                        self.handle_to_triples();
                    let terms_edge = self.handle_to_edge_triples();
                    Box::new(terms.chain(terms_edge))
                }
                Some(SubjectType::PathIri) => Box::new(self.paths()),
                Some(SubjectType::StepIri) => Box::new(self.steps()),
                Some(SubjectType::StepBorderIri) => {
                    Box::new(self.steps())
                }
                None => Box::new(Vec::new().into_iter()),
            };
            return terms;
        } else {
            return Box::new(iter::empty());
        }
    }

    fn get_term_type(&self, term: &EncodedTerm) -> Option<SubjectType> {
        if let EncodedTerm::NamedNode { iri_id: _, value } = term {
            let mut parts = value.split("/").collect::<Vec<_>>();
            parts.reverse();
            if parts[1] == "node" {
                return Some(SubjectType::NodeIri);
            } else if parts[3] == "path" && parts[1] == "step" {
                return Some(SubjectType::StepIri);
            } else if parts[3] == "path" && parts[1] == "position" {
                return Some(SubjectType::StepBorderIri);
            } else if parts[1] == "path" {
                return Some(SubjectType::PathIri);
            } else {
                return None;
            }
        } else {
            None
        }
    }

    fn type_triples<'a>(
        &'a self,
    ) -> impl Iterator<Item = EncodedQuad> + 'a {
        gen!({
        if self.is_vocab(self.object.as_ref(), vg::NODE) {
            for triple in self.nodes(self.subject.as_ref()) {
                yield_!(triple);
            }
        } else if self.is_vocab(self.object.as_ref(), vg::PATH) {
            for triple in self.paths() {
                yield_!(triple);
            }
        } else if self.is_step_associated_type() {
            for triple in self.steps() {
                yield_!(triple);
            }
        }
        }).into_iter()
    }

    fn empty_gen<'a>(&'a self) -> impl Iterator<Item = EncodedQuad> + 'a {
        gen!({}).into_iter()
    }

    fn nodes<'a>(
        &'a self,
        subject: Option<&'a EncodedTerm>,
    ) -> impl Iterator<Item = EncodedQuad> + 'a {
        gen!({
            match subject {
                Some(sub) => {
                    let is_node_iri = self.is_node_iri_in_graph(sub);
                    if self.is_vocab(self.predicate.as_ref(), rdf::TYPE)
                        && (self.is_vocab(self.object.as_ref(), vg::NODE) || self.object.is_none())
                        && is_node_iri
                    {
                        // println!("NF: type self.predicate");
                        yield_!(EncodedQuad::new(
                            sub.clone(),
                            rdf::TYPE.into(),
                            vg::NODE.into(),
                            self.graph_name.clone(),
                        ));
                    } else if self.predicate.is_none() && self.is_vocab(self.object.as_ref(), vg::NODE) && is_node_iri
                    {
                        // println!("NF: node self.object");
                        yield_!(EncodedQuad::new(
                            sub.clone(),
                            rdf::TYPE.into(),
                            vg::NODE.into(),
                            self.graph_name.clone(),
                        ));
                    } else if self.predicate.is_none() && is_node_iri {
                        // println!("NF: none self.predicate");
                        yield_!(EncodedQuad::new(
                            sub.clone(),
                            rdf::TYPE.into(),
                            vg::NODE.into(),
                            self.graph_name.clone(),
                        ));
                    }

                    if is_node_iri {
                        // println!("NF: is_node_iri");
                        for triple in self
                            .handle_to_triples()
                            .chain(self.handle_to_edge_triples())
                        {
                            yield_!(triple);
                        }
                    }
                }
                None => {
                    for handle in self.storage.graph.handles() {
                        let term = self
                            .handle_to_namednode(handle)
                            .expect("Can turn handle to namednode");
                        for triple in self.nodes(Some(&term)) {
                            yield_!(triple)
                        }
                    }
                }
            }
        })
        .into_iter()
    }

    fn paths<'a>(
        &'a self,
    ) -> impl Iterator<Item = EncodedQuad> + 'a {
        gen!({
            for triple in self
                .storage
                .graph
                .path_ids()
                .map(|path_id| {
                    let Some(path_name) = self.storage.graph.get_path_name(path_id) else {
                        return None;
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
                    None
                })
                .flatten()
            {
                yield_!(triple);
            }
        })
        .into_iter()
    }

    fn steps<'a>(
        &'a self,
    ) -> impl Iterator<Item = EncodedQuad> + 'a {
        gen!({
            if self.subject.is_none() {
                // println!("SF: none self.subject");
                for path_id in self.storage.graph.path_ids() {
                    if let Some(path_ref) = self.storage.graph.get_path_ref(path_id) {
                        let path_name = self.get_path_name(path_id).unwrap();
                        let mut rank = 1;
                        let mut position = 1;
                        let step_handle = path_ref.step_at(path_ref.first_step());
                        if step_handle.is_none() {
                            return;
                        }
                        let mut step_handle = step_handle.unwrap();
                        let mut node_handle = step_handle.handle();
                        for triple in self.step_handle_to_triples(
                            &path_name,
                            node_handle,
                            Some(rank),
                            Some(position),
                        ) {
                            yield_!(triple);
                        }

                        let steps = self
                            .storage
                            .graph
                            .path_steps(path_id)
                            .expect("Path has steps");
                        // TODO: for: does this make sense to be parallelized?
                        for _ in steps.skip(1) {
                            step_handle = path_ref.next_step(step_handle.0).unwrap();
                            position += self.storage.graph.node_len(node_handle);
                            node_handle = step_handle.handle();
                            rank += 1;
                            for triple in self.step_handle_to_triples(
                                &path_name,
                                node_handle,
                                Some(rank),
                                Some(position),
                            ) {
                                yield_!(triple);
                            }
                        }
                    }
                }
                for path_id in self.storage.graph.path_ids() {
                    if let Some(path_ref) = self.storage.graph.get_path_ref(path_id) {
                        let path_name = self.get_path_name(path_id).unwrap();
                        let mut rank = 1;
                        let mut position = 1;
                        let step_handle = path_ref.step_at(path_ref.first_step());
                        if step_handle.is_none() {
                            continue;
                        }
                        let mut step_handle = step_handle.unwrap();
                        let mut node_handle = step_handle.handle();
                        for triple in self.step_handle_to_triples(
                            &path_name,
                            node_handle,
                            Some(rank),
                            Some(position),
                        ) {
                            yield_!(triple);
                        }

                        let steps = self
                            .storage
                            .graph
                            .path_steps(path_id)
                            .expect("Path has steps");
                        // TODO: for: does this make sense to be parallelized?
                        for _ in steps.skip(1) {
                            step_handle = path_ref.next_step(step_handle.0).unwrap();
                            position += self.storage.graph.node_len(node_handle);
                            node_handle = step_handle.handle();
                            rank += 1;
                            for triple in self.step_handle_to_triples(
                                &path_name,
                                node_handle,
                                Some(rank),
                                Some(position),
                            ) {
                                yield_!(triple);
                            }
                        }
                    }
                }
            } else if let Some(step_type) = self.get_step_iri_fields(self.subject.as_ref()) {
                // println!("SF: some self.subject");
                match step_type {
                    StepType::Rank(path_name, target_rank) => {
                        // println!("RANK: {}, {}", path_name, target_rank);
                        if let Some(id) = self.storage.graph.get_path_id(path_name.as_bytes()) {
                            let path_ref = self.storage.graph.get_path_ref(id).unwrap();
                            let step_handle = path_ref.step_at(path_ref.first_step());
                            let mut step_handle = step_handle.unwrap();
                            let mut node_handle = step_handle.handle();
                            let mut rank = 1;
                            let mut position = 1;

                            let steps = self.storage.graph.path_steps(id).expect("Path has steps");
                            // TODO: for -> probably cannot/does not make sense be parallelized
                            for _ in steps.skip(1) {
                                if rank >= target_rank {
                                    break;
                                }
                                step_handle = path_ref.next_step(step_handle.0).unwrap();
                                position += self.storage.graph.node_len(node_handle);
                                node_handle = step_handle.handle();
                                rank += 1;
                            }
                            // println!("Now handling: {}, {}, {}", rank, position, node_handle.0);
                            for triple in self.step_handle_to_triples(
                                &path_name,
                                node_handle,
                                Some(rank),
                                Some(position),
                            ) {
                                yield_!(triple);
                            }
                            //results.append(&mut triples);
                        }
                    }
                    StepType::Position(path_name, position) => {
                        // println!("POSITION: {}, {}", path_name, position);
                        if let Some(id) = self.storage.graph.get_path_id(path_name.as_bytes()) {
                            if let Some(step) = self.storage.graph.path_step_at_base(id, position) {
                                let node_handle =
                                    self.storage.graph.path_handle_at_step(id, step).unwrap();
                                let rank = step.pack() as usize + 1;
                                for triple in self.step_handle_to_triples(
                                    &path_name,
                                    node_handle,
                                    Some(rank),
                                    Some(position),
                                ) {
                                    yield_!(triple);
                                }
                                //results.append(&mut triples);
                            }
                        }
                    }
                }
                //results
            }
        })
        .into_iter()
    }

    fn get_step_iri_fields(&self, term: Option<&EncodedTerm>) -> Option<StepType> {
        let term = term?;
        if let EncodedTerm::NamedNode { iri_id: _, value } = term {
            let mut parts = value.split("/").collect::<Vec<_>>();
            parts.reverse();
            if parts.len() < 5 || !parts.contains(&"path") {
                println!("We are quitting early! {:?}", parts);
                return None;
            }
            match parts[1] {
                "step" => {
                    let step_idx = value.rfind("step").expect("Should contain step");
                    let start = self.storage.base.len() + 1 + LEN_OF_PATH_AND_SLASH;
                    let path_text = &value[start..step_idx - 1].replace("/", "#");
                    let path_name = decode(&path_text).ok()?.to_string();
                    println!("Step: {}\t{}\t{}", step_idx, path_text, path_name);
                    Some(StepType::Rank(path_name, parts[0].parse().ok()?))
                }
                "position" => {
                    let pos_idx = value.rfind("position").expect("Should contain position");
                    let start = self.storage.base.len() + 1 + LEN_OF_PATH_AND_SLASH;
                    let path_text = &value[start..pos_idx - 1].replace("/", "#");
                    let path_name = decode(&path_text).ok()?.to_string();
                    println!("Pos: {}\t{}\t{}", pos_idx, path_text, path_name);
                    Some(StepType::Position(path_name, parts[0].parse().ok()?))
                }
                _ => None,
            }
        } else {
            None
        }
    }

    fn step_handle_to_triples<'a>(
        &'a self,
        path_name: &'a str,
        node_handle: Handle,
        rank: Option<usize>,
        position: Option<usize>,
    ) -> impl Iterator<Item = EncodedQuad> + 'a {
        gen!({
            let step_iri = self.step_to_namednode(path_name, rank).unwrap();
            let node_len = self.storage.graph.node_len(node_handle);
            let path_iri = self.path_to_namednode(path_name).unwrap();
            let rank = rank.unwrap() as i64;
            let position = position.unwrap() as i64;
            let position_literal = EncodedTerm::IntegerLiteral(position.into());
            // println!("SH");

            if self.subject.is_none() || step_iri == self.subject.as_ref().unwrap().clone() {
                if self.is_vocab(self.predicate.as_ref(), rdf::TYPE) || self.predicate.is_none() {
                    if self.object.is_none() || self.is_vocab(self.object.as_ref(), vg::STEP) {
                        // println!("SH: none/type self.predicate");
                        yield_!(EncodedQuad::new(
                            step_iri.clone(),
                            rdf::TYPE.into(),
                            vg::STEP.into(),
                            self.graph_name.clone(),
                        ));
                    }
                    if self.object.is_none() || self.is_vocab(self.object.as_ref(), faldo::REGION) {
                        // println!("SH: region self.object");
                        yield_!(EncodedQuad::new(
                            step_iri.clone(),
                            rdf::TYPE.into(),
                            faldo::REGION.into(),
                            self.graph_name.clone(),
                        ));
                    }
                }
                let node_iri = self.handle_to_namednode(node_handle).unwrap();
                if (self.is_vocab(self.predicate.as_ref(), vg::NODE_PRED)
                    || self.predicate.is_none() && !node_handle.is_reverse())
                    && (self.object.is_none() || node_iri == self.object.as_ref().unwrap().clone())
                {
                    // println!("SH: node self.object");
                    yield_!(EncodedQuad::new(
                        step_iri.clone(),
                        vg::NODE_PRED.into(),
                        node_iri.clone(),
                        self.graph_name.clone(),
                    ));
                }

                if (self.is_vocab(self.predicate.as_ref(), vg::REVERSE_OF_NODE)
                    || self.predicate.is_none() && node_handle.is_reverse())
                    && (self.object.is_none() || node_iri == self.object.as_ref().unwrap().clone())
                {
                    // println!("SH: reverse node self.object");
                    yield_!(EncodedQuad::new(
                        step_iri.clone(),
                        vg::REVERSE_OF_NODE.into(),
                        node_iri,
                        self.graph_name.clone(),
                    ));
                }

                if self.is_vocab(self.predicate.as_ref(), vg::RANK) || self.predicate.is_none() {
                    let rank_literal = EncodedTerm::IntegerLiteral(rank.into());
                    if self.object.is_none() || self.object.as_ref().unwrap().clone() == rank_literal {
                        // println!("SH: rank self.predicate");
                        yield_!(EncodedQuad::new(
                            step_iri.clone(),
                            vg::RANK.into(),
                            rank_literal,
                            self.graph_name.clone(),
                        ));
                    }
                }

                if self.is_vocab(self.predicate.as_ref(), vg::POSITION) || self.predicate.is_none() {
                    if self.object.is_none() || self.object.as_ref().unwrap().clone() == position_literal {
                        // println!("SH: position self.predicate");
                        yield_!(EncodedQuad::new(
                            step_iri.clone(),
                            vg::POSITION.into(),
                            position_literal.clone(),
                            self.graph_name.clone(),
                        ));
                    }
                }

                if self.is_vocab(self.predicate.as_ref(), vg::PATH_PRED) || self.predicate.is_none() {
                    if self.object.is_none() || path_iri == self.object.as_ref().unwrap().clone() {
                        // println!("SH: path self.predicate");
                        yield_!(EncodedQuad::new(
                            step_iri.clone(),
                            vg::PATH_PRED.into(),
                            path_iri.clone(),
                            self.graph_name.clone(),
                        ));
                    }
                }

                if self.predicate.is_none() || self.is_vocab(self.predicate.as_ref(), faldo::BEGIN) {
                    if self.object.is_none() || self.object.as_ref().unwrap().clone() == position_literal {
                        // println!("SH: begin self.predicate");
                        yield_!(EncodedQuad::new(
                            step_iri.clone(),
                            faldo::BEGIN.into(),
                            self.get_faldo_border_namednode(position as usize, path_name)
                                .unwrap(), // FIX
                            self.graph_name.clone(),
                        ));
                    }
                }
                if self.predicate.is_none() || self.is_vocab(self.predicate.as_ref(), faldo::END) {
                    // TODO: End position_literal vs position + node_len
                    if self.object.is_none() || self.object.as_ref().unwrap().clone() == position_literal {
                        // println!("SH: end self.predicate");
                        yield_!(EncodedQuad::new(
                            step_iri,
                            faldo::END.into(),
                            self.get_faldo_border_namednode(
                                position as usize + node_len,
                                path_name,
                            )
                            .unwrap(), // FIX
                            self.graph_name.clone(),
                        ));
                    }
                }

                if self.subject.is_none() {
                    // println!("SH: trailing none self.subject");
                    let begin_pos = position as usize;
                    let begin = self.get_faldo_border_namednode(begin_pos, path_name);
                    for triple in self.faldo_for_step(
                        begin_pos,
                        path_iri.clone(),
                        begin,
                    ) {
                        yield_!(triple);
                    }
                    let end_pos = position as usize + node_len;
                    let end = self.get_faldo_border_namednode(end_pos, path_name);
                    for triple in
                        self.faldo_for_step(end_pos, path_iri, end)
                    {
                        yield_!(triple);
                    }
                }
            }
        })
        .into_iter()
    }

    fn get_faldo_border_namednode(&self, position: usize, path_name: &str) -> Option<EncodedTerm> {
        let path_name = encode(path_name);
        let path_name = path_name.replace(URL_HASH, "/");
        let text = format!(
            "{}/path/{}/position/{}",
            self.storage.base, path_name, position
        );
        let named_node = NamedNode::new(text).unwrap();
        Some(named_node.as_ref().into())
    }

    fn faldo_for_step<'a>(
        &'a self,
        position: usize,
        path_iri: EncodedTerm,
        subject: Option<EncodedTerm>,
    ) -> impl Iterator<Item = EncodedQuad> + 'a {
        gen!({
            let ep = EncodedTerm::IntegerLiteral((position as i64).into());
            if (self.predicate.is_none() || self.is_vocab(self.predicate.as_ref(), faldo::POSITION_PRED))
                && (self.object.is_none() || self.object.as_ref() == Some(&ep))
            {
                // println!("FS: position");
                yield_!(EncodedQuad::new(
                    subject.clone().unwrap(),
                    faldo::POSITION_PRED.into(),
                    ep,
                    self.graph_name.clone(),
                ));
            }
            if (self.predicate.is_none() || self.is_vocab(self.predicate.as_ref(), rdf::TYPE))
                && (self.object.is_none() || self.is_vocab(self.object.as_ref(), faldo::EXACT_POSITION))
            {
                // println!("FS: position");
                yield_!(EncodedQuad::new(
                    subject.clone().unwrap(),
                    rdf::TYPE.into(),
                    faldo::EXACT_POSITION.into(),
                    self.graph_name.clone(),
                ));
            }
            if (self.predicate.is_none() || self.is_vocab(self.predicate.as_ref(), rdf::TYPE))
                && (self.object.is_none() || self.is_vocab(self.object.as_ref(), faldo::POSITION))
            {
                yield_!(EncodedQuad::new(
                    subject.clone().unwrap(),
                    rdf::TYPE.into(),
                    faldo::POSITION.into(),
                    self.graph_name.clone(),
                ));
            }
            if (self.predicate.is_none() || self.is_vocab(self.predicate.as_ref(), faldo::REFERENCE))
                && (self.object.is_none() || self.object.as_ref() == Some(&path_iri))
            {
                yield_!(EncodedQuad::new(
                    subject.unwrap(),
                    faldo::REFERENCE.into(),
                    path_iri,
                    self.graph_name.clone(),
                ));
            }
        })
        .into_iter()
    }

    fn handle_to_triples<'a>(
        &'a self,
    ) -> impl Iterator<Item = EncodedQuad> + 'a {
        gen!({
            if self.is_vocab(self.predicate.as_ref(), rdf::VALUE) || self.predicate.is_none() {
                let handle = Handle::new(
                    self.get_node_id(self.subject.as_ref().unwrap()).expect("Subject is node"),
                    Orientation::Forward,
                );
                let seq_bytes = self.storage.graph.sequence_vec(handle);
                let seq = str::from_utf8(&seq_bytes).expect("Node contains sequence");
                let seq_value = Literal::new_simple_literal(seq);
                if self.object.is_none()
                    || self.decode_term(self.object.as_ref().unwrap()).unwrap()
                        == Term::Literal(seq_value.clone())
                {
                    yield_!(EncodedQuad::new(
                        self.subject.clone().unwrap(),
                        rdf::VALUE.into(),
                        seq_value.as_ref().into(),
                        self.graph_name.clone(),
                    ));
                }
            }
        })
        .into_iter()
    }

    fn handle_to_edge_triples<'a>(
        &'a self,
    ) -> impl Iterator<Item = EncodedQuad> + 'a {
        gen!({
            if self.predicate.is_none() || self.is_node_related() {
                let handle = Handle::new(
                    self.get_node_id(self.subject.as_ref().unwrap()).expect("Subject has node id"),
                    Orientation::Forward,
                );
                for neighbor in self.storage.graph.neighbors(handle, Direction::Right) {
                    if self.object.is_none()
                        || self
                            .get_node_id(self.object.as_ref().unwrap())
                            .expect("Object has node id")
                            == neighbor.unpack_number()
                    {
                        for triple in
                            self.generate_edge_triples(handle, neighbor)
                        {
                            yield_!(triple);
                        }
                    }
                }
            }
        })
        .into_iter()
    }

    fn generate_edge_triples<'a>(
        &'a self,
        subject: Handle,
        object: Handle,
    ) -> impl Iterator<Item = EncodedQuad> + 'a {
        gen!({
            let node_is_reverse = subject.is_reverse();
            let other_is_reverse = object.is_reverse();
            if (self.predicate.is_none() || self.is_vocab(self.predicate.as_ref(), vg::LINKS_FORWARD_TO_FORWARD))
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
            if (self.predicate.is_none() || self.is_vocab(self.predicate.as_ref(), vg::LINKS_FORWARD_TO_REVERSE))
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
            if (self.predicate.is_none() || self.is_vocab(self.predicate.as_ref(), vg::LINKS_REVERSE_TO_FORWARD))
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
            if (self.predicate.is_none() || self.is_vocab(self.predicate.as_ref(), vg::LINKS_REVERSE_TO_REVERSE))
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
        let named_node = NamedNode::new(text).unwrap();
        Some(named_node.as_ref().into())
    }

    fn step_to_namednode(&self, path_name: &str, rank: Option<usize>) -> Option<EncodedTerm> {
        // println!("STEP_TO_NAMEDNODE: {} - {:?}", path_name, rank);
        let path_name = encode(path_name);
        let path_name = path_name.replace(URL_HASH, "/");
        let text = format!("{}/path/{}/step/{}", self.storage.base, path_name, rank?);
        let named_node = NamedNode::new(text).ok()?;
        Some(named_node.as_ref().into())
    }

    fn path_to_namednode(&self, path_name: &str) -> Option<EncodedTerm> {
        // println!("PATH_TO_NAMEDNODE: {}", path_name);
        let path_name = encode(path_name);
        let path_name = path_name.replace(URL_HASH, "/");
        let text = format!("{}/path/{}", self.storage.base, path_name);
        let named_node = NamedNode::new(text).ok()?;
        Some(named_node.as_ref().into())
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

impl Iterator for GraphIter {
    type Item = EncodedQuad;

    fn next(&mut self) -> Option<EncodedQuad> {
        None
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

// FIX: Change usize to u64
enum StepType {
    Rank(String, usize),
    Position(String, usize),
}

enum SubjectType {
    PathIri,
    StepBorderIri,
    NodeIri,
    StepIri,
}

// #[cfg(test)]
// mod tests {
//     use std::{path::Path, str::FromStr};
// 
//     use crate::storage::small_string::SmallString;
// 
//     // Note this useful idiom: importing names from outer (for mod tests) scope.
//     use super::*;
//     const BASE: &'static str = "https://example.org";
// 
//     fn _get_generator(gfa: &str) -> StorageGenerator {
//         let storage = Storage::from_str(gfa).unwrap();
//         StorageGenerator::new(storage)
//     }
// 
//     fn get_odgi_test_file_generator(file_name: &str) -> StorageGenerator {
//         let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(file_name);
//         let storage = Storage::open(&path).unwrap();
//         StorageGenerator::new(storage)
//     }
// 
//     fn print_quad(quad: &EncodedQuad) {
//         let sub = match &quad.subject {
//             EncodedTerm::NamedNode { iri_id: _, value } => value.clone(),
//             _ => "NOT NAMED".to_owned(),
//         };
//         let pre = match &quad.predicate {
//             EncodedTerm::NamedNode { iri_id: _, value } => value.clone(),
//             _ => "NOT NAMED".to_owned(),
//         };
//         let obj = match &quad.object {
//             EncodedTerm::NamedNode { iri_id: _, value } => value.clone(),
//             EncodedTerm::SmallStringLiteral(value) => format!("\"{}\"", value).to_string(),
//             EncodedTerm::IntegerLiteral(value) => value.to_string(),
//             _ => "NOT NAMED".to_owned(),
//         };
//         println!("{}\t{}\t{} .", sub, pre, obj);
//     }
// 
//     fn get_node(id: i64) -> EncodedTerm {
//         let text = format!("{}/node/{}", BASE, id);
//         let named_node = NamedNode::new(text).unwrap();
//         named_node.as_ref().into()
//     }
// 
//     fn get_step(path: &str, id: i64) -> EncodedTerm {
//         let path = encode(path);
//         let path = path.replace(URL_HASH, "/");
//         let text = format!("{}/path/{}/step/{}", BASE, path, id);
//         let named_node = NamedNode::new(text).unwrap();
//         named_node.as_ref().into()
//     }
// 
//     fn get_position(path: &str, id: i64) -> EncodedTerm {
//         let path = encode(path);
//         let path = path.replace(URL_HASH, "/");
//         let text = format!("{}/path/{}/position/{}", BASE, path, id);
//         let named_node = NamedNode::new(text).unwrap();
//         named_node.as_ref().into()
//     }
// 
//     fn get_path(path: &str) -> EncodedTerm {
//         let path = encode(path);
//         let path = path.replace(URL_HASH, "/");
//         let text = format!("{}/path/{}", BASE, path);
//         let named_node = NamedNode::new(text).unwrap();
//         named_node.as_ref().into()
//     }
// 
//     fn count_subjects(subject: &EncodedTerm, triples: &Vec<EncodedQuad>) -> usize {
//         let mut count = 0;
//         for triple in triples {
//             if &triple.subject == subject {
//                 count += 1;
//             }
//         }
//         count
//     }
// 
//     #[test]
//     fn test_single_node() {
//         let gen = get_odgi_test_file_generator("t_red.gfa");
//         let node_triple: Vec<_> = gen
//             .nodes(None, None, None, &EncodedTerm::DefaultGraph)
//             .collect();
//         let node_id_quad = EncodedQuad::new(
//             get_node(1),
//             rdf::TYPE.into(),
//             vg::NODE.into(),
//             EncodedTerm::DefaultGraph,
//         );
//         let sequence_quad = EncodedQuad::new(
//             get_node(1),
//             rdf::VALUE.into(),
//             EncodedTerm::SmallStringLiteral(SmallString::from_str("CAAATAAG").unwrap()),
//             EncodedTerm::DefaultGraph,
//         );
//         assert_eq!(node_triple.len(), 2);
//         assert!(node_triple.contains(&node_id_quad));
//         assert!(node_triple.contains(&sequence_quad));
//     }
// 
//     #[test]
//     fn test_single_node_type_spo() {
//         let gen = get_odgi_test_file_generator("t_red.gfa");
//         let node_1 = get_node(1);
//         let node_triple: Vec<_> = gen
//             .nodes(
//                 Some(&node_1),
//                 Some(&rdf::TYPE.into()),
//                 Some(&vg::NODE.into()),
//                 &EncodedTerm::DefaultGraph,
//             )
//             .collect();
//         let node_id_quad = EncodedQuad::new(
//             get_node(1),
//             rdf::TYPE.into(),
//             vg::NODE.into(),
//             EncodedTerm::DefaultGraph,
//         );
//         for tripe in &node_triple {
//             print_quad(tripe);
//         }
//         assert_eq!(node_triple.len(), 1);
//         assert!(node_triple.contains(&node_id_quad));
//     }
// 
//     #[test]
//     fn test_single_node_type_s() {
//         let gen = get_odgi_test_file_generator("t_red.gfa");
//         let node_triple: Vec<_> = gen
//             .nodes(Some(&get_node(1)), None, None, &EncodedTerm::DefaultGraph)
//             .collect();
//         let node_id_quad = EncodedQuad::new(
//             get_node(1),
//             rdf::TYPE.into(),
//             vg::NODE.into(),
//             EncodedTerm::DefaultGraph,
//         );
//         let sequence_quad = EncodedQuad::new(
//             get_node(1),
//             rdf::VALUE.into(),
//             EncodedTerm::SmallStringLiteral(SmallString::from_str("CAAATAAG").unwrap()),
//             EncodedTerm::DefaultGraph,
//         );
//         for tripe in &node_triple {
//             print_quad(tripe);
//         }
//         assert_eq!(node_triple.len(), 2);
//         assert!(node_triple.contains(&node_id_quad));
//         assert!(node_triple.contains(&sequence_quad));
//     }
// 
//     #[test]
//     fn test_single_node_type_p() {
//         let gen = get_odgi_test_file_generator("t_red.gfa");
//         let node_triple: Vec<_> = gen
//             .nodes(
//                 None,
//                 Some(&rdf::TYPE.into()),
//                 None,
//                 &EncodedTerm::DefaultGraph,
//             )
//             .collect();
//         let node_id_quad = EncodedQuad::new(
//             get_node(1),
//             rdf::TYPE.into(),
//             vg::NODE.into(),
//             EncodedTerm::DefaultGraph,
//         );
//         for tripe in &node_triple {
//             print_quad(tripe);
//         }
//         assert_eq!(node_triple.len(), 1);
//         assert!(node_triple.contains(&node_id_quad));
//     }
// 
//     #[test]
//     fn test_single_node_type_o() {
//         let gen = get_odgi_test_file_generator("t_red.gfa");
//         let node_triple: Vec<_> = gen
//             .nodes(
//                 None,
//                 None,
//                 Some(&vg::NODE.into()),
//                 &EncodedTerm::DefaultGraph,
//             )
//             .collect();
//         let node_id_quad = EncodedQuad::new(
//             get_node(1),
//             rdf::TYPE.into(),
//             vg::NODE.into(),
//             EncodedTerm::DefaultGraph,
//         );
//         for tripe in &node_triple {
//             print_quad(tripe);
//         }
//         assert_eq!(node_triple.len(), 1);
//         assert!(node_triple.contains(&node_id_quad));
//     }
// 
//     #[test]
//     fn test_double_node() {
//         // Reminder: fails with "old" version of rs-handlegraph (use git-master)
//         let gen = get_odgi_test_file_generator("t_double.gfa");
//         let node_triple: Vec<_> = gen
//             .nodes(None, None, None, &EncodedTerm::DefaultGraph)
//             .collect();
//         let links_quad = EncodedQuad::new(
//             get_node(1),
//             vg::LINKS.into(),
//             get_node(2),
//             EncodedTerm::DefaultGraph,
//         );
//         let links_f2f_quad = EncodedQuad::new(
//             get_node(1),
//             vg::LINKS_FORWARD_TO_FORWARD.into(),
//             get_node(2),
//             EncodedTerm::DefaultGraph,
//         );
//         for tripe in &node_triple {
//             print_quad(tripe);
//         }
//         assert_eq!(node_triple.len(), 6);
//         assert!(node_triple.contains(&links_quad));
//         assert!(node_triple.contains(&links_f2f_quad));
//     }
// 
//     #[test]
//     // TODO: Fix position numbers e.g. having pos/1 + pos/9 and pos/9 + pos/10
//     fn test_step() {
//         let gen = get_odgi_test_file_generator("t_step.gfa");
//         let step_triples: Vec<_> = gen
//             .steps(None, None, None, &EncodedTerm::DefaultGraph)
//             .collect();
//         for triple in &step_triples {
//             print_quad(triple);
//         }
//         let count_step1 = count_subjects(&get_step("x#a", 1), &step_triples);
//         let count_step2 = count_subjects(&get_step("x#a", 2), &step_triples);
//         let count_pos1 = count_subjects(&get_position("x#a", 1), &step_triples);
//         let count_pos9 = count_subjects(&get_position("x#a", 9), &step_triples);
//         let count_pos10 = count_subjects(&get_position("x#a", 10), &step_triples);
//         assert_eq!(count_step1, 8, "Number of step 1 triples");
//         assert_eq!(count_step2, 8, "Number of step 2 triples");
//         assert_eq!(count_pos1, 4, "Number of pos 1 triples");
//         assert_eq!(count_pos9, 8, "Number of pos 9 triples");
//         assert_eq!(count_pos10, 4, "Number of pos 10 triples");
//     }
// 
//     #[test]
//     fn test_step_s() {
//         let gen = get_odgi_test_file_generator("t_step.gfa");
//         let step_triples: Vec<_> = gen
//             .steps(
//                 Some(&get_step("x#a", 1)),
//                 None,
//                 None,
//                 &EncodedTerm::DefaultGraph,
//             )
//             .collect();
//         for triple in &step_triples {
//             print_quad(triple);
//         }
//         assert_eq!(step_triples.len(), 8, "Number of step 1 triples");
//     }
// 
//     #[test]
//     fn test_step_p() {
//         let gen = get_odgi_test_file_generator("t_step.gfa");
//         let step_triples: Vec<_> = gen
//             .steps(
//                 None,
//                 Some(&rdf::TYPE.into()),
//                 None,
//                 &EncodedTerm::DefaultGraph,
//             )
//             .collect();
//         for triple in &step_triples {
//             print_quad(triple);
//         }
//         assert_eq!(step_triples.len(), 12, "Number of type triples");
//     }
// 
//     #[test]
//     fn test_step_o() {
//         let gen = get_odgi_test_file_generator("t_step.gfa");
//         let step_triples: Vec<_> = gen
//             .steps(None, None, Some(&get_node(1)), &EncodedTerm::DefaultGraph)
//             .collect();
//         for triple in &step_triples {
//             print_quad(triple);
//         }
//         assert_eq!(step_triples.len(), 1, "Number of type triples");
//     }
// 
//     #[test]
//     fn test_step_node() {
//         let gen = get_odgi_test_file_generator("t.gfa");
//         let step_triples: Vec<_> = gen
//             .steps(
//                 None,
//                 Some(&vg::NODE_PRED.into()),
//                 None,
//                 &EncodedTerm::DefaultGraph,
//             )
//             .collect();
//         for triple in &step_triples {
//             print_quad(triple);
//         }
//         let quad = EncodedQuad::new(
//             get_step("x#a", 6),
//             vg::NODE_PRED.into(),
//             get_node(9),
//             EncodedTerm::DefaultGraph,
//         );
//         assert_eq!(step_triples.len(), 10, "Number of node_pred triples");
//         assert!(step_triples.contains(&quad));
//     }
// 
//     #[test]
//     fn test_paths() {
//         let gen = get_odgi_test_file_generator("t.gfa");
//         let generic_triples: Vec<_> = gen
//             .paths(None, None, None, &EncodedTerm::DefaultGraph)
//             .collect();
//         let specific_triples: Vec<_> = gen
//             .paths(
//                 Some(&get_path("x#a")),
//                 Some(&rdf::TYPE.into()),
//                 Some(&vg::PATH.into()),
//                 &EncodedTerm::DefaultGraph,
//             )
//             .collect();
//         for triple in &generic_triples {
//             print_quad(triple)
//         }
//         let quad = EncodedQuad::new(
//             get_path("x#a"),
//             rdf::TYPE.into(),
//             vg::PATH.into(),
//             EncodedTerm::DefaultGraph,
//         );
//         assert_eq!(generic_triples, specific_triples);
//         assert_eq!(generic_triples.len(), 1);
//         assert!(generic_triples.contains(&quad));
//     }
// 
//     #[test]
//     fn test_full() {
//         let gen = get_odgi_test_file_generator("t.gfa");
//         let node_triple = gen.quads_for_pattern(None, None, None, &EncodedTerm::DefaultGraph);
//         //for tripe in &node_triple.first.terms {
//         //    print_quad(tripe);
//        // }
//         assert_eq!(1, 1);
//     }
// }
