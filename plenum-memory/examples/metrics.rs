//! Scores the cases tools/check_metrics.py writes, for comparison with NLTK and rouge_score.

use plenum_memory::metrics;

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let cases: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&a[0]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let list = |k: &str| cases[k].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect::<Vec<_>>();
    let (preds, refs, words) = (list("preds"), list("refs"), list("words"));
    let (b1, b4) = metrics::bleu(&preds, &refs);
    let (r1, rl) = metrics::rouge(&preds, &refs);
    let out = serde_json::json!({
        "stems": words.iter().map(|w| metrics::porter_stem(w)).collect::<Vec<_>>(),
        "bleu1": b1, "bleu4": b4, "rouge1": r1, "rougeL": rl,
    });
    std::fs::write(&a[1], out.to_string()).map_err(|e| e.to_string())?;
    Ok(())
}
