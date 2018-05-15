#[macro_use]
extern crate log;
extern crate byteorder;
extern crate env_logger;

use byteorder::{ByteOrder, NativeEndian};
use std::str;

pub struct XGBModel {
    num_features: usize,
    trees: Vec<XGBTree>,
}

struct XGBTree {
    nodes: Vec<XGBNode>,
}

struct XGBNode {
    cleft: i32, // node is leaf when cleft == -1
    split_index: u32,
    cdefault: i32,
    node_value: f32,
    cright: i32,
}

pub struct FVec {
    data: Vec<f32>,
}

impl FVec {
    pub fn new(num_features: usize) -> FVec {
        FVec {
            data: vec![0.0; num_features],
        }
    }

    #[inline]
    fn is_misssing(&self, index: usize) -> bool {
        self.data[index] == 0.0
    }

    #[inline]
    fn fvalue(&self, index: usize) -> f32 {
        self.data[index]
    }

    pub fn set(&mut self, indices: &[u32], values: &[f32]) {
        let length = indices.len();
        for i in 0..length {
            self.data[indices[i] as usize] = values[i];
        }
    }

    pub fn reset(&mut self, indices: &[u32]) {
        for i in indices {
            self.data[*i as usize] = 0.0;
        }
    }
}

impl XGBModel {
    pub fn load(buffer: &[u8]) -> Option<XGBModel> {
        // >>> LearnerModelParam
        // float base_score
        let mut offset = 0;
        let base_score = NativeEndian::read_f32(&buffer[offset..]);
        offset += 4;
        debug!("base_score:{:.6}", base_score);
        // int padding[33]
        offset += 4 * 33;
        // <<< LearnerModelParam

        //uint64_t len
        let name_obj_str_len = NativeEndian::read_u64(&buffer[offset..]) as usize;
        offset += 8;
        debug!("name_obj_str_len:{}", name_obj_str_len);

        // string name_obj
        let name_obj = str::from_utf8(&buffer[offset..(offset + name_obj_str_len)]).unwrap();
        offset += name_obj_str_len;
        debug!("name_obj:{}", name_obj);

        //uint64_t len
        let name_gbm_str_len = NativeEndian::read_u64(&buffer[offset..]) as usize;
        offset += 8;
        debug!("name_gbm_str_len:{}", name_gbm_str_len);

        // string name_gbm
        let name_gbm = str::from_utf8(&buffer[offset..(offset + name_gbm_str_len)]).unwrap();
        offset += name_gbm_str_len;
        debug!("name_gbm:{}", name_gbm);

        // >>> GBTreeModelParam
        // int num_trees
        let num_trees = NativeEndian::read_i32(&buffer[offset..]);
        if num_trees <= 0 {
            return None;
        }
        offset += 4;
        debug!("num_trees:{}", num_trees);
        offset += 4;

        // int num_features
        let num_features = NativeEndian::read_i32(&buffer[offset..]);
        offset += 4;
        if num_features <= 0 {
            return None;
        }
        debug!("num_features:{}", num_features);

        offset += 4 + 8;
        // int num_output_group
        let num_output_group = NativeEndian::read_i32(&buffer[offset..]);
        offset += 4;
        debug!("num_output_group:{}", num_output_group);

        offset += 4 * 33;
        // <<< GBTreeModelParam

        let mut trees = Vec::with_capacity(num_trees as usize);
        for _ in 0..num_trees {
            let (length, tree) = XGBTree::load(&buffer[offset..]);
            trees.push(tree);
            offset += length;
        }

        Some(XGBModel {
            num_features: num_features as usize,
            trees,
        })
    }

    #[inline]
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    #[inline]
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn predict_leaf(&self, feats: &FVec, tree_limit: usize, preds: &mut [u32]) {
        let tree_limit = if tree_limit <= 0 {
            self.num_trees()
        } else {
            tree_limit
        };

        for i in 0..tree_limit {
            let tree = &self.trees[i];
            let nid = tree.get_leaf_index(feats);
            preds[i] = nid as u32;
        }
    }

    pub fn predict_value(&self, feats: &FVec, mut tree_limit: usize) -> f32 {
        if tree_limit <= 0 {
            tree_limit = self.num_trees();
        }

        let mut value = 0.0;
        for i in 0..tree_limit {
            let tree = &self.trees[i];
            let nid = tree.get_leaf_index(feats);
            value += tree.get_leaf_value(nid);
        }
        value
    }
}

impl XGBTree {
    fn load(buffer: &[u8]) -> (usize, XGBTree) {
        let mut offset = 0;

        // >>> TreeParam
        offset += 4;
        // int num_nodes
        let num_nodes = NativeEndian::read_i32(&buffer[offset..]);
        offset += 4;
        debug!("num_nodes:{}", num_nodes);

        offset += 4 * 3;
        let size_leaf_vector = NativeEndian::read_i32(&buffer[offset..]);
        offset += 4;
        debug!("size_leaf_vector:{}", size_leaf_vector);
        offset += 4 * 31;
        // <<< TreeParam

        // >>> nodes
        let mut nodes = Vec::with_capacity(num_nodes as usize);
        for _ in 0..num_nodes {
            let (length, node) = XGBNode::load(&buffer[offset..]);
            nodes.push(node);
            offset += length;
        }
        // >>> nodes

        // CONSUME_BYTES(fi, (3 * sizeof(bst_float) + sizeof(int)) * param.num_nodes);
        offset += (4 * 4) * (num_nodes as usize);

        if size_leaf_vector != 0 {
            let dummy_len = NativeEndian::read_u64(&buffer[offset..]);
            debug!("dummy_len:{}", dummy_len);
            offset += 8;
            if dummy_len > 0 {
                offset += 4 * (dummy_len as usize);
            }
        }
        return (offset, XGBTree { nodes });
    }

    #[inline]
    fn get_leaf_index(&self, feats: &FVec) -> usize {
        let mut nid = 0;
        let mut node = &self.nodes[nid];
        while !node.is_leaf() {
            let split_index = node.get_split_index();
            if feats.is_misssing(split_index as usize) {
                nid = node.get_cdefault() as usize;
            } else {
                let split_cond = node.get_split_cond();
                if feats.fvalue(split_index as usize) < split_cond {
                    nid = node.get_cleft() as usize;
                } else {
                    nid = node.get_cright() as usize;
                }
            }
            node = &self.nodes[nid];
        }
        nid
    }

    #[inline]
    fn get_leaf_value(&self, nid: usize) -> f32 {
        self.nodes[nid].get_leaf_value()
    }
}

impl XGBNode {
    pub fn load(buffer: &[u8]) -> (usize, XGBNode) {
        let mut offset = 0;
        let parent = NativeEndian::read_i32(&buffer[offset..]);
        offset += 4;
        debug!("parent:{}", parent);
        let cleft = NativeEndian::read_i32(&buffer[offset..]);
        offset += 4;
        debug!("cleft:{}", cleft);
        let cright = NativeEndian::read_i32(&buffer[offset..]);
        offset += 4;
        debug!("cright:{}", cright);
        let split_index = NativeEndian::read_u32(&buffer[offset..]);
        offset += 4;
        debug!("split_index:{}", split_index);

        let node_value = if cleft == -1 {
            let leaf_value = NativeEndian::read_f32(&buffer[offset..]);
            debug!("leaf_value:{:.6}", leaf_value);
            leaf_value
        } else {
            let split_cond = NativeEndian::read_f32(&buffer[offset..]);
            debug!("split_cond:{:.6}", split_cond);
            split_cond
        };
        offset += 4;
        let cdefault = if (split_index >> 31) != 0 {
            cleft
        } else {
            cright
        };
        let split_index = split_index & ((1 << 31) - 1);
        (
            offset,
            XGBNode {
                split_index,
                cdefault,
                cleft,
                cright,
                node_value,
            },
        )
    }

    #[inline]
    fn is_leaf(&self) -> bool {
        self.cleft == -1
    }

    #[inline]
    fn get_split_index(&self) -> u32 {
        self.split_index
    }

    #[inline]
    fn get_split_cond(&self) -> f32 {
        self.node_value
    }

    #[inline]
    fn get_leaf_value(&self) -> f32 {
        self.node_value
    }

    #[inline]
    fn get_cright(&self) -> i32 {
        self.cright
    }

    #[inline]
    fn get_cleft(&self) -> i32 {
        self.cleft
    }
    #[inline]
    fn get_cdefault(&self) -> i32 {
        self.cdefault
    }
}
