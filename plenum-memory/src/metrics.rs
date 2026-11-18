//! The scores of Memorilla's evaluation (port of memorilla/metrics.py): exact match,
//! yes/no/maybe accuracy, token F1, corpus BLEU-1/4 with NLTK's method-1 smoothing, and
//! ROUGE-1/L F-measure with rouge_score's tokenizer and NLTK's Porter stemmer. Answer-
//! choice accuracy needs the encoder and lives in `eval.rs`. METEOR is not ported (it
//! needs WordNet's synonym data).

use fancy_regex::Regex;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Removes chat-template tokens and surrounding whitespace.
pub fn clean_generation(text: &str) -> String {
    let mut t = text.to_string();
    for tok in ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "<|im_sep|>"] {
        t = t.replace(tok, "");
    }
    t.trim().to_string()
}

/// Lower-cased whitespace tokens (Python's `text.lower().split()`).
fn ws_tokens(text: &str) -> Vec<String> {
    text.to_lowercase().split_whitespace().map(String::from).collect()
}

/// SQuAD answer normalisation: lower case, no ASCII punctuation, no articles, single spaces.
pub fn normalize_answer(text: &str) -> String {
    static ARTICLES: OnceLock<Regex> = OnceLock::new();
    let re = ARTICLES.get_or_init(|| Regex::new(r"\b(a|an|the)\b").unwrap());
    let t: String = text.to_lowercase().chars().filter(|c| !c.is_ascii_punctuation()).collect();
    let t = re.replace_all(&t, " ");
    t.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn exact_match(preds: &[String], refs: &[String]) -> f64 {
    mean(preds.iter().zip(refs).map(|(p, r)| (normalize_answer(p) == normalize_answer(r)) as u8 as f64))
}

/// The leading yes / no / maybe of a text, past up to three list markers or quotes.
pub fn first_polarity(text: &str) -> Option<&'static str> {
    let mut t = text.trim();
    for _ in 0..3 {
        if t.is_empty() {
            return None;
        }
        if t.starts_with(['-', '*', '•']) {
            t = t[t.chars().next().unwrap().len_utf8()..].trim_start();
        } else if t.starts_with(['"', '\'']) {
            t = &t[1..];
        } else if t.starts_with("1.") || t.starts_with("2.") || t.starts_with("3.") {
            t = t[2..].trim_start();
        } else {
            break;
        }
    }
    let head = t.split_whitespace().next()?.trim_end_matches(['.', ',', '!', '?', ':', ';', '"', '\'']).to_lowercase();
    ["yes", "no", "maybe"].into_iter().find(|w| *w == head)
}

/// Accuracy over the rows whose reference starts with yes / no / maybe.
pub fn polarity_accuracy(preds: &[String], refs: &[String]) -> f64 {
    let pairs: Vec<bool> = preds.iter().zip(refs).filter_map(|(p, r)| first_polarity(r).map(|g| first_polarity(p) == Some(g))).collect();
    if pairs.is_empty() {
        0.0
    } else {
        pairs.iter().filter(|b| **b).count() as f64 / pairs.len() as f64
    }
}

fn counts(toks: &[String]) -> HashMap<&str, usize> {
    let mut m = HashMap::new();
    for t in toks {
        *m.entry(t.as_str()).or_insert(0) += 1;
    }
    m
}

/// Mean word-overlap F1 on lower-cased whitespace tokens.
pub fn token_f1(preds: &[String], refs: &[String]) -> f64 {
    mean(preds.iter().zip(refs).map(|(p, r)| {
        let (pt, gt) = (ws_tokens(p), ws_tokens(r));
        if pt.is_empty() || gt.is_empty() {
            return 0.0;
        }
        let (pc, gc) = (counts(&pt), counts(&gt));
        let overlap: usize = pc.iter().map(|(t, n)| (*n).min(*gc.get(t).unwrap_or(&0))).sum();
        if overlap == 0 {
            return 0.0;
        }
        let (pr, rc) = (overlap as f64 / pt.len() as f64, overlap as f64 / gt.len() as f64);
        2.0 * pr * rc / (pr + rc)
    }))
}

fn ngram_counts(t: &[String], n: usize) -> HashMap<&[String], usize> {
    let mut m = HashMap::new();
    if t.len() >= n {
        for w in t.windows(n) {
            *m.entry(w).or_insert(0) += 1;
        }
    }
    m
}

/// Corpus BLEU-1 and BLEU-4 as NLTK's `corpus_bleu` with `SmoothingFunction().method1`
/// computes them: clipped n-gram counts summed over the corpus (a hypothesis shorter
/// than n adds 1 to the denominator), epsilon 0.1 for a zero numerator, and the
/// brevity penalty on the summed lengths.
pub fn bleu(preds: &[String], refs: &[String]) -> (f64, f64) {
    let (mut num, mut den) = ([0usize; 5], [0usize; 5]);
    let (mut hyp_len, mut ref_len) = (0usize, 0usize);
    for (p, r) in preds.iter().zip(refs) {
        let (h, g) = (ws_tokens(p), ws_tokens(r));
        for n in 1..=4 {
            let hc = ngram_counts(&h, n);
            let gc = ngram_counts(&g, n);
            num[n] += hc.iter().map(|(k, c)| (*c).min(*gc.get(k).unwrap_or(&0))).sum::<usize>();
            den[n] += hc.values().sum::<usize>().max(1);
        }
        hyp_len += h.len();
        ref_len += g.len();
    }
    if num[1] == 0 {
        return (0.0, 0.0);
    }
    let bp = if hyp_len > ref_len {
        1.0
    } else if hyp_len == 0 {
        0.0
    } else {
        (1.0 - ref_len as f64 / hyp_len as f64).exp()
    };
    let p: Vec<f64> = (1..=4).map(|n| if num[n] == 0 { 0.1 / den[n] as f64 } else { num[n] as f64 / den[n] as f64 }).collect();
    let b1 = bp * p[0].ln().exp();
    let b4 = bp * (0.25 * p.iter().map(|x| x.ln()).sum::<f64>()).exp();
    (b1, b4)
}

/// rouge_score's tokenizer: lower case, runs of anything but [a-z0-9] become spaces,
/// tokens longer than 3 characters Porter-stemmed.
fn rouge_tokens(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let cleaned: String = lower.chars().map(|c| if c.is_ascii_lowercase() || c.is_ascii_digit() { c } else { ' ' }).collect();
    cleaned.split_whitespace().map(|t| if t.chars().count() > 3 { porter_stem(t) } else { t.to_string() }).filter(|t| !t.is_empty()).collect()
}

fn f_measure(p: f64, r: f64) -> f64 {
    if p + r > 0.0 {
        2.0 * p * r / (p + r)
    } else {
        0.0
    }
}

fn lcs(a: &[String], b: &[String]) -> usize {
    let mut prev = vec![0usize; b.len() + 1];
    let mut cur = vec![0usize; b.len() + 1];
    for x in a {
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = if x == y { prev[j] + 1 } else { cur[j].max(prev[j + 1]) };
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Mean ROUGE-1 and ROUGE-L F-measure (rouge_score, use_stemmer=True).
pub fn rouge(preds: &[String], refs: &[String]) -> (f64, f64) {
    let mut r1 = Vec::new();
    let mut rl = Vec::new();
    for (p, r) in preds.iter().zip(refs) {
        let (pt, gt) = (rouge_tokens(p.trim()), rouge_tokens(r));
        if pt.is_empty() || gt.is_empty() {
            r1.push(0.0);
            rl.push(0.0);
            continue;
        }
        let (pc, gc) = (counts(&pt), counts(&gt));
        let overlap: usize = pc.iter().map(|(t, n)| (*n).min(*gc.get(t).unwrap_or(&0))).sum();
        r1.push(f_measure(overlap as f64 / pt.len() as f64, overlap as f64 / gt.len() as f64));
        let l = lcs(&gt, &pt) as f64;
        rl.push(f_measure(l / pt.len() as f64, l / gt.len() as f64));
    }
    (mean(r1.into_iter()), mean(rl.into_iter()))
}

fn mean(it: impl Iterator<Item = f64>) -> f64 {
    let (mut s, mut n) = (0.0, 0usize);
    for x in it {
        s += x;
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        s / n as f64
    }
}

// ----------------------------------------------------------------------------------
// NLTK's Porter stemmer, mode NLTK_EXTENSIONS (the default, which rouge_score uses).
// ----------------------------------------------------------------------------------

fn is_cons(w: &[u8], i: usize) -> bool {
    match w[i] {
        b'a' | b'e' | b'i' | b'o' | b'u' => false,
        b'y' => i == 0 || !is_cons(w, i - 1),
        _ => true,
    }
}

fn measure(w: &[u8]) -> usize {
    let mut seq = Vec::with_capacity(w.len());
    for i in 0..w.len() {
        seq.push(if is_cons(w, i) { b'c' } else { b'v' });
    }
    seq.windows(2).filter(|p| p == b"vc").count()
}

fn has_vowel(w: &[u8]) -> bool {
    (0..w.len()).any(|i| !is_cons(w, i))
}

fn ends_double_cons(w: &[u8]) -> bool {
    let n = w.len();
    n >= 2 && w[n - 1] == w[n - 2] && is_cons(w, n - 1)
}

fn ends_cvc(w: &[u8]) -> bool {
    let n = w.len();
    (n >= 3 && is_cons(w, n - 3) && !is_cons(w, n - 2) && is_cons(w, n - 1) && !matches!(w[n - 1], b'w' | b'x' | b'y'))
        || (n == 2 && !is_cons(w, 0) && is_cons(w, 1))
}

type Cond<'a> = Option<&'a dyn Fn(&[u8]) -> bool>;

/// `_apply_rule_list`: the first rule whose suffix matches decides; if its condition
/// fails the word is returned unchanged. A "*d" suffix means a double consonant.
fn apply_rules(w: &[u8], rules: &[(&str, &[u8], Cond)]) -> Vec<u8> {
    for (suffix, rep, cond) in rules {
        if *suffix == "*d" {
            if ends_double_cons(w) {
                let stem = &w[..w.len() - 2];
                return if cond.is_none_or(|c| c(stem)) { [stem, rep].concat() } else { w.to_vec() };
            }
            continue;
        }
        if w.ends_with(suffix.as_bytes()) {
            let stem = &w[..w.len() - suffix.len()];
            return if cond.is_none_or(|c| c(stem)) { [stem, rep].concat() } else { w.to_vec() };
        }
    }
    w.to_vec()
}

fn m_gt0(s: &[u8]) -> bool {
    measure(s) > 0
}
fn m_gt1(s: &[u8]) -> bool {
    measure(s) > 1
}

fn step1a(w: &[u8]) -> Vec<u8> {
    if w.ends_with(b"ies") && w.len() == 4 {
        return [&w[..w.len() - 3], b"ie"].concat();
    }
    apply_rules(w, &[("sses", b"ss", None), ("ies", b"i", None), ("ss", b"ss", None), ("s", b"", None)])
}

fn step1b(w: &[u8]) -> Vec<u8> {
    if w.ends_with(b"ied") {
        return [&w[..w.len() - 3], if w.len() == 4 { b"ie".as_slice() } else { b"i".as_slice() }].concat();
    }
    if w.ends_with(b"eed") {
        let stem = &w[..w.len() - 3];
        return if measure(stem) > 0 { [stem, b"ee"].concat() } else { w.to_vec() };
    }
    let mut inter: Option<&[u8]> = None;
    for suffix in [b"ed".as_slice(), b"ing".as_slice()] {
        if w.ends_with(suffix) {
            let s = &w[..w.len() - suffix.len()];
            if has_vowel(s) {
                inter = Some(s);
                break;
            }
        }
    }
    let Some(st) = inter else { return w.to_vec() };
    let last = st.last().copied().unwrap_or(b'?');
    let keep_double = move |_: &[u8]| !matches!(last, b'l' | b's' | b'z');
    let add_e = |s: &[u8]| measure(s) == 1 && ends_cvc(s);
    let last_b = [last];
    apply_rules(st, &[("at", b"ate", None), ("bl", b"ble", None), ("iz", b"ize", None), ("*d", &last_b, Some(&keep_double)), ("", b"e", Some(&add_e))])
}

fn step1c(w: &[u8]) -> Vec<u8> {
    let cond = |s: &[u8]| s.len() > 1 && is_cons(s, s.len() - 1);
    apply_rules(w, &[("y", b"i", Some(&cond))])
}

fn step2(w: &[u8]) -> Vec<u8> {
    if w.ends_with(b"alli") && m_gt0(&w[..w.len() - 4]) {
        return step2(&[&w[..w.len() - 4], b"al"].concat());
    }
    let logi = |_: &[u8]| m_gt0(&w[..w.len() - 3]);
    let m: Cond = Some(&m_gt0);
    apply_rules(
        w,
        &[
            ("ational", b"ate", m),
            ("tional", b"tion", m),
            ("enci", b"ence", m),
            ("anci", b"ance", m),
            ("izer", b"ize", m),
            ("bli", b"ble", m),
            ("alli", b"al", m),
            ("entli", b"ent", m),
            ("eli", b"e", m),
            ("ousli", b"ous", m),
            ("ization", b"ize", m),
            ("ation", b"ate", m),
            ("ator", b"ate", m),
            ("alism", b"al", m),
            ("iveness", b"ive", m),
            ("fulness", b"ful", m),
            ("ousness", b"ous", m),
            ("aliti", b"al", m),
            ("iviti", b"ive", m),
            ("biliti", b"ble", m),
            ("fulli", b"ful", m),
            ("logi", b"log", Some(&logi)),
        ],
    )
}

fn step3(w: &[u8]) -> Vec<u8> {
    let m: Cond = Some(&m_gt0);
    apply_rules(w, &[("icate", b"ic", m), ("ative", b"", m), ("alize", b"al", m), ("iciti", b"ic", m), ("ical", b"ic", m), ("ful", b"", m), ("ness", b"", m)])
}

fn step4(w: &[u8]) -> Vec<u8> {
    let m: Cond = Some(&m_gt1);
    let ion = |s: &[u8]| measure(s) > 1 && matches!(s.last(), Some(b's') | Some(b't'));
    apply_rules(
        w,
        &[
            ("al", b"", m),
            ("ance", b"", m),
            ("ence", b"", m),
            ("er", b"", m),
            ("ic", b"", m),
            ("able", b"", m),
            ("ible", b"", m),
            ("ant", b"", m),
            ("ement", b"", m),
            ("ment", b"", m),
            ("ent", b"", m),
            ("ion", b"", Some(&ion)),
            ("ou", b"", m),
            ("ism", b"", m),
            ("ate", b"", m),
            ("iti", b"", m),
            ("ous", b"", m),
            ("ive", b"", m),
            ("ize", b"", m),
        ],
    )
}

fn step5a(w: &[u8]) -> Vec<u8> {
    if let Some(stem) = w.strip_suffix(b"e") {
        let m = measure(stem);
        if m > 1 || (m == 1 && !ends_cvc(stem)) {
            return stem.to_vec();
        }
    }
    w.to_vec()
}

fn step5b(w: &[u8]) -> Vec<u8> {
    let cond = |_: &[u8]| measure(&w[..w.len() - 1]) > 1;
    apply_rules(w, &[("ll", b"l", Some(&cond))])
}

/// NLTK's `PorterStemmer().stem(word)` for a lower-case ASCII word.
pub fn porter_stem(word: &str) -> String {
    let w = word.to_lowercase();
    let irregular = match w.as_str() {
        "skies" | "sky" => Some("sky"),
        "dying" => Some("die"),
        "lying" => Some("lie"),
        "tying" => Some("tie"),
        "news" => Some("news"),
        "innings" | "inning" => Some("inning"),
        "outings" | "outing" => Some("outing"),
        "cannings" | "canning" => Some("canning"),
        "howe" => Some("howe"),
        "proceed" => Some("proceed"),
        "exceed" => Some("exceed"),
        "succeed" => Some("succeed"),
        _ => None,
    };
    if let Some(s) = irregular {
        return s.to_string();
    }
    if w.chars().count() <= 2 || !w.is_ascii() {
        return w;
    }
    let mut s = w.into_bytes();
    for step in [step1a, step1b, step1c, step2, step3, step4, step5a, step5b] {
        s = step(&s);
    }
    String::from_utf8(s).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn porter_known_words() {
        for (w, want) in [
            ("caresses", "caress"),
            ("ponies", "poni"),
            ("ties", "tie"),
            ("cats", "cat"),
            ("feed", "feed"),
            ("agreed", "agre"),
            ("plastered", "plaster"),
            ("motoring", "motor"),
            ("sing", "sing"),
            ("conflated", "conflat"),
            ("troubled", "troubl"),
            ("sized", "size"),
            ("hopping", "hop"),
            ("falling", "fall"),
            ("filing", "file"),
            ("happy", "happi"),
            ("relational", "relat"),
            ("conditional", "condit"),
            ("generalization", "gener"),
            ("running", "run"),
            ("dying", "die"),
        ] {
            assert_eq!(porter_stem(w), want, "{w}");
        }
    }

    #[test]
    fn simple_metrics() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert_eq!(exact_match(&s(&["The Eiffel Tower!"]), &s(&["eiffel tower"])), 1.0);
        assert_eq!(first_polarity("- \"Yes. It does.\""), Some("yes"));
        assert_eq!(first_polarity("Possibly"), None);
        assert!((token_f1(&s(&["a b c"]), &s(&["a b d e"])) - 2.0 * (2.0 / 3.0) * 0.5 / (2.0 / 3.0 + 0.5)).abs() < 1e-12);
        assert_eq!(clean_generation("  hi<|im_end|>"), "hi");
    }
}
