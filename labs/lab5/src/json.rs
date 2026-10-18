//! Provided: a minimal JSON reader (std only, offline). Enough for Mooncake trace lines
//! and the R1 cost-model file; not a general-purpose parser (no \u escapes beyond ASCII).

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn parse(text: &str) -> Result<Json, String> {
        let b = text.as_bytes();
        let mut pos = 0;
        let value = parse_value(b, &mut pos)?;
        skip_ws(b, &mut pos);
        if pos != b.len() {
            return Err(format!("trailing data at byte {pos}"));
        }
        Ok(value)
    }
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(items) => items.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn num(&self) -> Option<f64> {
        match self {
            Json::Num(x) => Some(*x),
            _ => None,
        }
    }
    pub fn str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(v) => Some(v),
            _ => None,
        }
    }
}

fn skip_ws(b: &[u8], pos: &mut usize) {
    while *pos < b.len() && b[*pos].is_ascii_whitespace() {
        *pos += 1;
    }
}

fn parse_value(b: &[u8], pos: &mut usize) -> Result<Json, String> {
    skip_ws(b, pos);
    let Some(&c) = b.get(*pos) else {
        return Err("unexpected end of input".into());
    };
    match c {
        b'{' => {
            *pos += 1;
            let mut items = Vec::new();
            skip_ws(b, pos);
            if b.get(*pos) == Some(&b'}') {
                *pos += 1;
                return Ok(Json::Obj(items));
            }
            loop {
                skip_ws(b, pos);
                let key = match parse_value(b, pos)? {
                    Json::Str(s) => s,
                    _ => return Err(format!("object key must be a string at byte {pos}")),
                };
                skip_ws(b, pos);
                if b.get(*pos) != Some(&b':') {
                    return Err(format!("expected ':' at byte {pos}"));
                }
                *pos += 1;
                items.push((key, parse_value(b, pos)?));
                skip_ws(b, pos);
                match b.get(*pos) {
                    Some(b',') => *pos += 1,
                    Some(b'}') => {
                        *pos += 1;
                        return Ok(Json::Obj(items));
                    }
                    _ => return Err(format!("expected ',' or '}}' at byte {pos}")),
                }
            }
        }
        b'[' => {
            *pos += 1;
            let mut items = Vec::new();
            skip_ws(b, pos);
            if b.get(*pos) == Some(&b']') {
                *pos += 1;
                return Ok(Json::Arr(items));
            }
            loop {
                items.push(parse_value(b, pos)?);
                skip_ws(b, pos);
                match b.get(*pos) {
                    Some(b',') => *pos += 1,
                    Some(b']') => {
                        *pos += 1;
                        return Ok(Json::Arr(items));
                    }
                    _ => return Err(format!("expected ',' or ']' at byte {pos}")),
                }
            }
        }
        b'"' => {
            *pos += 1;
            let mut out = String::new();
            while let Some(&ch) = b.get(*pos) {
                *pos += 1;
                match ch {
                    b'"' => return Ok(Json::Str(out)),
                    b'\\' => {
                        let esc = *b.get(*pos).ok_or("bad escape")?;
                        *pos += 1;
                        out.push(match esc {
                            b'n' => '\n',
                            b't' => '\t',
                            b'"' => '"',
                            b'\\' => '\\',
                            b'/' => '/',
                            _ => return Err(format!("unsupported escape at byte {pos}")),
                        });
                    }
                    _ => {
                        // Copy one UTF-8 sequence starting at ch.
                        let start = *pos - 1;
                        let mut end = *pos;
                        while end < b.len() && (b[end] & 0xC0) == 0x80 {
                            end += 1;
                        }
                        out.push_str(
                            std::str::from_utf8(&b[start..end]).map_err(|e| e.to_string())?,
                        );
                        *pos = end;
                    }
                }
            }
            Err("unterminated string".into())
        }
        b't' if b[*pos..].starts_with(b"true") => {
            *pos += 4;
            Ok(Json::Bool(true))
        }
        b'f' if b[*pos..].starts_with(b"false") => {
            *pos += 5;
            Ok(Json::Bool(false))
        }
        b'n' if b[*pos..].starts_with(b"null") => {
            *pos += 4;
            Ok(Json::Null)
        }
        _ => {
            let start = *pos;
            while *pos < b.len()
                && matches!(b[*pos], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
            {
                *pos += 1;
            }
            let s = std::str::from_utf8(&b[start..*pos]).map_err(|e| e.to_string())?;
            s.parse::<f64>()
                .map(Json::Num)
                .map_err(|_| format!("bad value {:?} at byte {start}", s))
        }
    }
}
