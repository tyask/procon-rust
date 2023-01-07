use std::{path::PathBuf, io::{Write, BufWriter, BufReader, BufRead, Read}, error::Error, fs, collections::VecDeque};
use id_tree::*;
use itertools::Itertools;
use regex::Regex;

pub struct CargoCapture {
    module_project: PathBuf,
}

impl CargoCapture {
    pub fn new(module_project_path: &str) -> CargoCapture { CargoCapture { module_project: PathBuf::from(module_project_path) }}

    pub fn capture<R:Read>(&self, r: BufReader<R>) -> Result<String, Box<dyn Error>> {
        let mut tree = Tree::new();
        let root_id = tree.insert(Node::new(Token::Root), InsertBehavior::AsRoot).unwrap();
        self.parse_to(r, &mut tree, &root_id).unwrap();
        while let Some(_) = self.capture_one(&mut tree)? { }
        self.compress_modules(&mut tree);
        self.to_string(&tree, &root_id)
    }

    fn parse_to<R:Read>(&self, r: BufReader<R>, tree: &mut Tree<Token>, parent: &NodeId) -> Result<NodeId, Box<dyn Error>> {
        // CAP(ex::unionfind)
        let re_cap = Regex::new(r"\s*//\s*CAP\s*\((?P<MOD>.*?)\)").unwrap();
        // mod abc
        let re_mod = Regex::new(r"(?P<TOKEN>(pub\s)?\s*mod\s+(?P<MOD>\w+)\s*\{\s*)").unwrap();

        let root_id = tree.insert(Node::new(Token::Root), InsertBehavior::UnderNode(parent)).unwrap().clone();
        let mut curr_node_id = root_id.clone();
        let mut st_mod = VecDeque::<(String, i32)>::new();
        for (i, line) in r.lines().map(|r| r.unwrap()).enumerate() {
            if let Some(cap) = re_cap.captures(&line) {
                // キャプチャ行
                tree.insert_under(Token::Capture(line.to_owned(), cap["MOD"].to_owned()), &curr_node_id)?;
            } else if re_mod.is_match(&line) {
                // モジュール定義行
                // モジュール定義行にはモジュール定義以外のトークンは来ないことを前提とする.
                for cap in re_mod.captures_iter(&line) {
                    let module = cap["MOD"].to_owned();
                    curr_node_id = tree.insert_under(
                        Token::Module(cap["TOKEN"].to_owned(), module.to_owned()), &curr_node_id)?;
                    st_mod.push_back((module.to_owned(), 1));
                }
            } else {
                // 普通の行
                let mut moduled_closed = false;
                for c in line.chars() {
                    if st_mod.is_empty() { break; }
                    let d = if c == '{' { 1 } else if c == '}' { -1 } else { 0 };
                    if d == 0 { continue; }
                    if let Some((_, cnt)) = st_mod.back_mut() {
                        *cnt += d;
                        if *cnt == 0 {
                            curr_node_id = tree.get(&curr_node_id).unwrap().parent().unwrap().clone();
                            st_mod.pop_back();
                            moduled_closed = true;
                        } else if *cnt < 0 {
                            return Err(Box::new(CapError::CapError(format!("Inconsistent braces (line {})", i))));
                        }
                    }
                }

                // moduleをクローズしている行は消えるので注意.
                if !moduled_closed {
                    tree.insert_under(Token::Src(line.to_owned()), &curr_node_id)?;
                }
            }
        }
        Ok(root_id)

    }

    fn capture_one(&self, tree: &mut Tree<Token>) -> Result<Option<NodeId>, Box<dyn Error>> {
        let cap_id = tree.traverse_pre_order_ids(tree.root_node_id().unwrap()).unwrap()
            .find(|id| matches!(tree.get(id).map(|n|n.data()).unwrap(), Token::Capture(..)))
            ;
        if cap_id.is_none() { return Ok(None); }
        let cap_id = cap_id.unwrap();
        if let Token::Capture(_, module) = tree.get(&cap_id).unwrap().data().clone() {
            // キャプチャ行自体はRootにして出力されないようにする
            tree.get_mut(&cap_id).unwrap().replace_data(Token::Root);
            
            // モジュールを展開
            self.parse_from_module_to(&module, tree).and_then(|id| Ok(Some(id)))
        } else {
            unreachable!()
        }

    }

    fn parse_from_module_to(&self, module: &str, tree: &mut Tree<Token>) -> Result<NodeId, Box<dyn Error>> {

        // 展開先を探索し、なければ作成する.
        let mut parent = tree.root_node_id().cloned().unwrap();
        for m in module.split("::") {
            if let Some(id) = tree.lookup_module_under(m, &parent) {
                parent = id;
            } else {
                parent = tree.insert_under(Token::Module(format!("pub mod {} {{", m), m.to_owned()), &parent)?;
            }
        }

        if !tree.get(&parent).unwrap().children().is_empty() {
            // 重複して展開されないように、対象モジュールが既に存在する場合は一度削除して再度展開する.
            let d = tree.get(&parent).unwrap().data().clone();
            let n = tree.insert_under(d, &tree.root_node_id().unwrap().clone())?;
            tree.swap_nodes(&parent, &n, SwapBehavior::TakeChildren)?;
            tree.remove_node(parent, RemoveBehavior::DropChildren)?;
            parent = n;
        }

        let path = self.module_project.join("src").join(module.replace("::", "/") + ".rs");
        let s = BufReader::new(fs::read_to_string(path)?.as_bytes()).lines()
                .map(|r| r.unwrap() + "\n")
                .take_while(|s| !s.starts_with("// CAP(IGNORE_BELOW)"))
                .collect::<String>();
        let root_id = self.parse_to(BufReader::new(s.as_bytes()), tree, &parent)?;
        Ok(root_id)
    }

    // Token::Moduleが2つ以上続き、子にModuleしかないようなModuleは1行にまとめる
    fn compress_modules(&self, tree: &mut Tree<Token>) {
        let mut t = Vec::new();
        let mut m = Vec::new();
        for id in tree.traverse_pre_order_ids(tree.root_node_id().unwrap()).unwrap() {
            let n = tree.get(&id).unwrap();
            if n.data().is_module()
                    && n.children().iter().all(|c| tree.get(c).unwrap().data().is_module()) {
                t.push(id.clone());
            } else {
                if t.len() >= 2 { m.push(t.clone()); }
                t.clear();
            }
        }

        for t in &m {
            let texts = t.iter()
                .map(|id| tree.get(id).unwrap().data())
                .map(|d| match d { Token::Module(line, _) => line, _ => unreachable!() } )
                .cloned()
                .collect_vec();
            tree.get_mut(&t[0]).unwrap().replace_data(Token::Modules(texts));
            t.iter().skip(1).for_each(|id| { tree.remove_node(id.clone(), RemoveBehavior::LiftChildren).unwrap(); });
        }
    }

    fn to_string(&self, tree: &Tree<Token>, id: &NodeId) -> Result<String, Box<dyn Error>> {
        let mut buf = BufWriter::new(Vec::new());
        self.write_to(&mut buf, tree, id)?;
        Ok(String::from_utf8(buf.into_inner()?)?)
    }

    fn write_to<W: Write>(&self, buf: &mut BufWriter<W>, tree: &Tree<Token>, id: &NodeId) -> Result<(), Box<dyn Error>> {
        let n = tree.get(id).unwrap();
        match n.data() {
            Token::Src(line)       => { writeln!(buf, "{}", line)?; },
            Token::Module(line, _) => { writeln!(buf, "{}", line)?; },
            Token::Modules(texts)  => { writeln!(buf, "{}", texts.iter().join(" "))?; },
            _ => {},
        }

        for child_id in n.children() { self.write_to(buf, tree, child_id)?; }

        match n.data() {
            Token::Module(_, _)   => { writeln!(buf, "}}")?; },
            Token::Modules(texts) => { writeln!(buf, "{}", "}".repeat(texts.len()))?; }
            _ => {}
        }

        Ok(())
    }

}

#[derive(PartialEq, PartialOrd, Debug, Clone)]
enum Token {
    Root,
    Src(String),                 // text
    Module(String, String),      // text, module
    Modules(Vec<String>),        // texts
    Capture(String, String),     // text, capture
}

impl Token {
    fn is_module(&self) -> bool { matches!(self, Token::Module(_,_)) }
}

#[derive(thiserror::Error, Debug)]
enum CapError {
    #[error("Capture Error: `{0}`")]
    CapError(String),
}

trait TreeTrait<T> {
    fn find_parent_module(&self, module: &str) -> NodeId;
    fn lookup_module_under(&self, module: &str, parent: &NodeId) -> Option<NodeId>;
    fn insert_under(&mut self, t: Token, parent: &NodeId) -> Result<NodeId, NodeIdError>;
    fn to_debug_string(&self) -> Result<String, Box<dyn Error>>;
}

impl TreeTrait<Token> for Tree<Token> {
    fn find_parent_module(&self, module: &str) -> NodeId {
        let mods = module.split("::").collect_vec();
        let mut id = self.root_node_id().cloned().unwrap();
        for m in mods.iter().take(mods.len()-1) {
            if let Some(next) = self.lookup_module_under(m, &id) {
                id = next;
            } else {
                return id;
            }
        }
        id
    }

    fn lookup_module_under(&self, module: &str, parent: &NodeId) -> Option<NodeId> {
        self.traverse_pre_order_ids(parent).unwrap()
            .find(|id| {
                if let Token::Module(_, m) = self.get(id).unwrap().data() {
                    module == m
                } else {
                    false
                }
            })
    }

    fn insert_under(&mut self, t: Token, parent: &NodeId) -> Result<NodeId, NodeIdError> {
        self.insert(Node::new(t), InsertBehavior::UnderNode(parent))
    }

    fn to_debug_string(&self) -> Result<String, Box<dyn Error>> {
        let mut s = String::new();
        self.write_formatted(&mut s)?;
        Ok(s)
    }
}


#[cfg(test)]
mod tests {
    use std::io::BufReader;

    use super::CargoCapture;

    #[test]
    fn test_capture() {
        let content =
r#"aaa
pub mod a {
    xxx
}
// CAP(capture::test::mod1)
// CAP(capture::test::mod2)
bbb
"#.to_string();

        let expected =
r#"aaa
pub mod a {
    xxx
}
bbb
pub mod capture { pub mod test {
pub mod mod1 {
fn hello1() -> String { "Hello".to_string() }
}
pub mod mod2 {
fn hello2() -> String { "Hello".to_string() }
}
}}
"#.to_string();

        let cap = CargoCapture::new(".");
        let s = cap.capture(BufReader::new(content.as_bytes())).unwrap();
        assert_eq!(s, expected);
    }

    #[test]
    fn test_capture_in_module_file() {
        let content =
r#"aaa
// CAP(capture::test::mod3)
// CAP(capture::test::mod3)
bbb
"#.to_string();

        let expected =
r#"aaa
bbb
pub mod capture { pub mod test {
pub mod mod3 {
fn hello3() -> String { "Hello".to_string() }
}
pub mod mod2 {
fn hello2() -> String { "Hello".to_string() }
}
}}
"#.to_string();

        let cap = CargoCapture::new(".");
        let s = cap.capture(BufReader::new(content.as_bytes())).unwrap();
        println!("{}", s);
        assert_eq!(s, expected);
    }

}
