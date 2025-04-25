import streamlit as st
import yaml
import re
import requests

st.set_page_config(page_title="Pipeline Migrator", layout="wide")

# ---- STYLING ----
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    .css-1aumxhk {
        background-color: #1e1e2f;
        color: white;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown("### <img src='https://img.icons8.com/ios-filled/50/code-file.png' width='24'/> Pipeline Migrator", unsafe_allow_html=True)

def convert_github_to_raw(url):
    """
    Convert a standard GitHub URL to a raw GitHub URL.
    Example:
    https://github.com/user/repo/blob/main/Jenkinsfile
    -> https://raw.githubusercontent.com/user/repo/main/Jenkinsfile
    """
    if "github.com" in url and "/blob/" in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob", "")
    return url


col1, col2 = st.columns([1, 1])

# ---- FILE OR GITHUB INPUT ----
with col1:
    st.markdown("#### 📂 Provide Jenkinsfile")

    input_method = st.radio("Choose input method:", ["Upload File", "Import from GitHub"], horizontal=True)
    jenkins_code = ""

    if input_method == "Upload File":
        with st.container(border=True):
            st.markdown("💡 **Drag & Drop** your `.groovy` Jenkinsfile here or click to upload:")
            uploaded_file = st.file_uploader(
                label="Upload Jenkinsfile",
                type=["groovy"],
                label_visibility="collapsed"
            )

            if uploaded_file:
                jenkins_code = uploaded_file.read().decode("utf-8")

    elif input_method == "Import from GitHub":
        github_url = st.text_input("🔗 Enter the raw GitHub URL of the Jenkinsfile", placeholder="https://raw.githubusercontent.com/user/repo/branch/Jenkinsfile")

        if github_url:
            raw_url = convert_github_to_raw(github_url)
        try:
            response = requests.get(raw_url)
            if response.status_code == 200:
                jenkins_code = response.text
                st.success("✅ Successfully fetched Jenkinsfile from GitHub.")
            else:
                st.error(f"❌ Failed to fetch file. HTTP {response.status_code}")
        except Exception as e:
            st.error(f"❌ Error fetching file: {e}")


# ---- INPUT DISPLAY ----
col1.subheader("Input Jenkinsfile")

if jenkins_code:
    col1.code(jenkins_code, language="groovy")
else:
    col1.markdown("⚠️ No Jenkinsfile provided yet.")
    col1.code("// Upload or import a Jenkinsfile to view its contents here...", language="groovy")

# ---- OUTPUT HARNESS YAML ----
col2.subheader("Generated Harness YAML")

# Dummy YAML conversion
harness_yaml = {
    # Your future YAML logic goes here
}

harness_yaml_str = yaml.dump(harness_yaml, sort_keys=False)
col2.code(harness_yaml_str if jenkins_code else "# Converted YAML will appear here...", language="yaml")

col2.download_button(
    label="📥 Download YAML",
    data=harness_yaml_str,
    file_name="converted_pipeline.yaml",
    mime="text/yaml",
    disabled=(not jenkins_code)
)

# ---- VALIDATION RULE ENGINE ----
st.markdown("### ✅ Validation Results")

def validate_jenkins_code(code):
    results = []

    # Rule 1: Look for pipeline or node
    if "pipeline" in code or "node" in code:
        results.append("✅ Found `pipeline` or `node` block.")
    else:
        results.append("❌ Missing `pipeline` or `node` block.")

    # Rule 2: At least one stage
    if re.search(r"stage\s*\(", code):
        results.append("✅ Found at least one `stage` block.")
    else:
        results.append("❌ No `stage` blocks found.")

    # Rule 3: Check for steps
    if re.search(r"steps\s*\{", code):
        results.append("✅ Found `steps` block inside stage.")
    else:
        results.append("⚠️ `steps` block is missing or outside of `stage`.")

    # Rule 4: Agent declaration
    if "agent" in code:
        results.append("✅ Found `agent` declaration.")
    else:
        results.append("⚠️ No `agent` specified — default is `any`.")

    return results

if jenkins_code:
    validation_results = validate_jenkins_code(jenkins_code)
    for res in validation_results:
        st.markdown(f"- {res}")
else:
    st.info("Upload or import a Jenkinsfile to see validation results.")


# ---- FOOTER ----
st.markdown("""---""")
st.markdown("© 2025 Pipeline Migrator. All rights reserved.", unsafe_allow_html=True)
