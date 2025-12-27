use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let docs_out = Path::new(&out_dir).join("docs");
    fs::create_dir_all(&docs_out).unwrap();

    // Process each markdown file in docs/
    for entry in fs::read_dir("docs").unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.extension().is_some_and(|ext| ext == "md") {
            let content = fs::read_to_string(&path).unwrap();

            // Transform relative markdown links to rustdoc intra-doc links
            // Pattern: [text](./foo-bar.md) → [text](crate::guides::foo_bar)
            let transformed = transform_doc_links(&content);

            let out_path = docs_out.join(path.file_name().unwrap());
            fs::write(&out_path, transformed).unwrap();
        }
    }

    // Tell Cargo to rerun if any docs change
    println!("cargo:rerun-if-changed=docs/");
}

fn transform_doc_links(content: &str) -> String {
    let mut result = String::with_capacity(content.len());
    let mut remaining = content;

    while let Some(start) = remaining.find("](./") {
        // Copy everything before the link target
        result.push_str(&remaining[..start + 2]); // include "]("

        // Find the end of the link
        let after_prefix = &remaining[start + 4..]; // skip "](./""
        if let Some(end) = after_prefix.find(".md)") {
            let filename = &after_prefix[..end];
            // Convert kebab-case to snake_case
            let module_name = filename.replace('-', "_");
            result.push_str("crate::guides::");
            result.push_str(&module_name);
            result.push(')');
            remaining = &after_prefix[end + 4..]; // skip ".md)"
        } else {
            // Not a .md link, keep as-is
            result.push_str("./");
            remaining = after_prefix;
        }
    }

    result.push_str(remaining);
    result
}
