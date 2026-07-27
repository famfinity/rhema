// Validate the last-token pooling + Qwen3 prompt fix: paraphrase queries should rank the
// correct verse highest.
use rhema_detection::OnnxEmbedder;
use rhema_detection::semantic::embedder::TextEmbedder;
fn cos(a:&[f32],b:&[f32])->f32{ a.iter().zip(b).map(|(x,y)|x*y).sum() }
fn main(){
  let m="../models/qwen3-fe/model.onnx"; let t="../models/qwen3-fe/tokenizer.json";
  let mut e=OnnxEmbedder::load(std::path::Path::new(m),std::path::Path::new(t)).unwrap();
  // passages: NO prompt (Qwen3 document convention)
  e.set_prompt_prefix(String::new());
  let passages=[
    ("1Cor.13.4","Charity suffereth long, and is kind; charity envieth not"),
    ("John.3.16","For God so loved the world, that he gave his only begotten Son"),
    ("Gen.1.1","In the beginning God created the heaven and the earth"),
    ("Ps.23.1","The LORD is my shepherd; I shall not want"),
  ];
  let pv: Vec<(&str,Vec<f32>)>=passages.iter().map(|(r,tx)|(*r,e.embed(tx).unwrap())).collect();
  // queries: Qwen3 Instruct prompt
  e.set_prompt_prefix("Instruct: Given a spoken phrase, retrieve the Bible verse it quotes or refers to\nQuery:");
  for q in ["love is patient love is kind it does not envy","God loved everyone so much he gave his only son"]{
    let qv=e.embed(q).unwrap();
    let mut s: Vec<_>=pv.iter().map(|(r,v)|(*r,cos(&qv,v))).collect();
    s.sort_by(|a,b|b.1.partial_cmp(&a.1).unwrap());
    println!("\nQUERY: {q}");
    for (r,c) in s { println!("   {:>10}  cos={:.3}", r, c); }
  }
}
