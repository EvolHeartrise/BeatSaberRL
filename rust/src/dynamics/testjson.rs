//! Minimal JSON reader for the oracle refdata files (test-only; the
//! crate has no deps by house rule). Supports exactly what
//! gen_reference.py emits: objects, arrays, strings (no escapes beyond
//! \" \\ \/ \n \t), f64 numbers, bools, null.

#[derive(Debug, Clone)]
#[allow(dead_code)] // Bool payload: parsed for completeness, unread
pub enum Json {
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
    Bool(bool),
    Null,
}

impl Json {
    pub fn get(&self, key: &str) -> &Json {
        match self {
            Json::Obj(kv) => kv
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
                .unwrap_or_else(|| panic!("missing key {key}")),
            _ => panic!("get({key}) on non-object"),
        }
    }
    pub fn num(&self) -> f64 {
        match self {
            Json::Num(x) => *x,
            _ => panic!("not a number"),
        }
    }
    pub fn str(&self) -> &str {
        match self {
            Json::Str(s) => s,
            _ => panic!("not a string"),
        }
    }
    pub fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(v) => v,
            _ => panic!("not an array"),
        }
    }
    /// Array of numbers → Vec<f64>.
    pub fn nums(&self) -> Vec<f64> {
        self.arr().iter().map(|x| x.num()).collect()
    }
    /// Array of arrays of numbers → Vec<Vec<f64>>.
    pub fn rows(&self) -> Vec<Vec<f64>> {
        self.arr().iter().map(|r| r.nums()).collect()
    }
    /// Array of strings → Vec<String>.
    pub fn strs(&self) -> Vec<String> {
        self.arr().iter().map(|x| x.str().to_string()).collect()
    }
}

pub fn parse(text: &str) -> Json {
    let b = text.as_bytes();
    let mut pos = 0usize;
    let v = parse_value(b, &mut pos);
    skip_ws(b, &mut pos);
    assert_eq!(pos, b.len(), "trailing junk at byte {pos}");
    v
}

fn skip_ws(b: &[u8], pos: &mut usize) {
    while *pos < b.len() && matches!(b[*pos], b' ' | b'\t' | b'\n' | b'\r') {
        *pos += 1;
    }
}

fn parse_value(b: &[u8], pos: &mut usize) -> Json {
    skip_ws(b, pos);
    match b[*pos] {
        b'{' => {
            *pos += 1;
            let mut kv = Vec::new();
            skip_ws(b, pos);
            if b[*pos] == b'}' {
                *pos += 1;
                return Json::Obj(kv);
            }
            loop {
                skip_ws(b, pos);
                let k = parse_string(b, pos);
                skip_ws(b, pos);
                assert_eq!(b[*pos], b':');
                *pos += 1;
                let v = parse_value(b, pos);
                kv.push((k, v));
                skip_ws(b, pos);
                match b[*pos] {
                    b',' => *pos += 1,
                    b'}' => {
                        *pos += 1;
                        return Json::Obj(kv);
                    }
                    c => panic!("bad object separator {c} at {pos}"),
                }
            }
        }
        b'[' => {
            *pos += 1;
            let mut v = Vec::new();
            skip_ws(b, pos);
            if b[*pos] == b']' {
                *pos += 1;
                return Json::Arr(v);
            }
            loop {
                v.push(parse_value(b, pos));
                skip_ws(b, pos);
                match b[*pos] {
                    b',' => *pos += 1,
                    b']' => {
                        *pos += 1;
                        return Json::Arr(v);
                    }
                    c => panic!("bad array separator {c} at {pos}"),
                }
            }
        }
        b'"' => Json::Str(parse_string(b, pos)),
        b't' => {
            assert_eq!(&b[*pos..*pos + 4], b"true");
            *pos += 4;
            Json::Bool(true)
        }
        b'f' => {
            assert_eq!(&b[*pos..*pos + 5], b"false");
            *pos += 5;
            Json::Bool(false)
        }
        b'n' => {
            assert_eq!(&b[*pos..*pos + 4], b"null");
            *pos += 4;
            Json::Null
        }
        _ => {
            let start = *pos;
            while *pos < b.len()
                && matches!(b[*pos], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
            {
                *pos += 1;
            }
            let s = std::str::from_utf8(&b[start..*pos]).unwrap();
            Json::Num(s.parse::<f64>().unwrap_or_else(|_| panic!("bad number {s}")))
        }
    }
}

fn parse_string(b: &[u8], pos: &mut usize) -> String {
    assert_eq!(b[*pos], b'"', "expected string at {pos}");
    *pos += 1;
    let mut s = String::new();
    loop {
        match b[*pos] {
            b'"' => {
                *pos += 1;
                return s;
            }
            b'\\' => {
                *pos += 1;
                let c = b[*pos];
                s.push(match c {
                    b'"' => '"',
                    b'\\' => '\\',
                    b'/' => '/',
                    b'n' => '\n',
                    b't' => '\t',
                    _ => panic!("unsupported escape \\{}", c as char),
                });
                *pos += 1;
            }
            c => {
                s.push(c as char);
                *pos += 1;
            }
        }
    }
}
