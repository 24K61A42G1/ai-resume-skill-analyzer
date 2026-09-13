import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI-Based Resume Skill Analyzer")
st.write("Upload your resume, select or add a target job role, and get instant skill matching feedback.")

# File uploader widget
uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["txt", "pdf"])

resume_text = ""
if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        resume_text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(uploaded_file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    resume_text += text
        except ImportError:
            st.error("Please install pypdf (`pip install pypdf`) to parse PDF files.")

# Resume text preview toggle
if resume_text.strip():
    with st.expander("🔍 Preview Extracted Resume Text"):
        st.write(resume_text)

# Default job roles dictionary
job_roles = {
    "Data Scientist": ["python", "sql", "machine learning", "deep learning", "statistics", "data analysis", "pandas", "numpy"],
    "ML Engineer": ["python", "machine learning", "deep learning", "pytorch", "tensorflow", "nlp", "docker", "mlops"],
    "Software Engineer": ["python", "java", "c++", "data structures", "algorithms", "sql", "git", "oops"],
    "Frontend Developer": ["html", "css", "javascript", "react", "angular", "typescript", "tailwind", "git"],
    "Backend Developer": ["python", "node.js", "express", "sql", "mongodb", "apis", "docker", "git"],
    "Full Stack Developer": ["html", "css", "javascript", "react", "node.js", "python", "sql", "git"],
    "MERN Stack Developer": ["mongodb", "express", "react", "node.js", "javascript", "tailwind", "rest apis", "git"],
    "Data Analyst": ["excel", "sql", "python", "tableau", "power bi", "statistics", "pandas", "data visualization"],
    "Cybersecurity Analyst": ["networking", "linux", "security", "python", "cryptography", "wireshark", "firewalls", "siem"],
    "Cloud Engineer": ["aws", "azure", "gcp", "docker", "kubernetes", "linux", "terraform", "ci/cd"],
    "DevOps Engineer": ["docker", "kubernetes", "jenkins", "terraform", "aws", "linux", "ci/cd", "python"],
    "Android Developer": ["kotlin", "java", "android studio", "jetpack compose", "git", "firebase", "xml", "apis"],
    "UI/UX Designer": ["figma", "adobe xd", "wireframing", "prototyping", "user research", "ui design", "ux principles", "html"]
}

role_options = list(job_roles.keys()) + ["➕ Add Custom Role"]
selected_option = st.selectbox("Select Target Job Role:", role_options)

if selected_option == "➕ Add Custom Role":
    custom_role_name = st.text_input("Enter Custom Job Role Name:")
    custom_skills_input = st.text_input("Enter Required Skills (comma-separated):")
    if custom_role_name and custom_skills_input:
        required_skills = [s.strip().lower() for s in custom_skills_input.split(",")]
        selected_role = custom_role_name
    else:
        required_skills = []
else:
    selected_role = selected_option
    required_skills = job_roles[selected_role]

# Learning resource suggestions for missing skills
learning_resources = {
    "python": "Check official Python docs or freeCodeCamp Python courses.",
    "sql": "Practice queries on LeetCode or Mode Analytics SQL Tutorial.",
    "machine learning": "Explore Andrew Ng's Machine Learning Specialization on Coursera.",
    "deep learning": "Look into fast.ai or DeepLearning.AI courses.",
    "react": "Build projects using the official React documentation.",
    "node.js": "Explore Node.js crash courses and backend tutorials on YouTube.",
    "docker": "Read Docker Get Started documentation and tutorials.",
    "kubernetes": "Check Kubernetes official tutorials and Katacoda labs.",
    "aws": "Review AWS Skill Builder free learning plans."
}

if st.button("Analyze Resume"):
    if not resume_text.strip():
        st.warning("Please upload a valid resume file before analyzing.")
    elif not required_skills:
        st.warning("Please specify a valid role and required skills.")
    else:
        processed_resume = resume_text.lower()
        processed_resume = re.sub(r'[^a-z\s]', '', processed_resume)

        found = []
        missing = []

        for skill in required_skills:
            if skill in processed_resume:
                found.append(skill)
            else:
                missing.append(skill)

        rule_based_match = (len(found) / len(required_skills)) * 100

        job_text = " ".join(required_skills)
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([processed_resume, job_text])
        similarity_score = cosine_similarity(vectors)[0][1] * 100

        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
        col1.metric("Rule-Based Match", f"{rule_based_match:.2f}%")
        col2.metric("AI Similarity Score", f"{similarity_score:.2f}%")

        st.write("**Skills Found:**")
        st.write(found if found else "None")

        st.write("**Skills Missing:**")
        if missing:
            st.write(missing)
            st.subheader("📚 Learning Recommendations for Missing Skills")
            for m_skill in missing:
                suggestion = learning_resources.get(m_skill, f"Search tutorials or documentation online for '{m_skill}'.")
                st.info(f"**{m_skill.capitalize()}**: {suggestion}")
        else:
            st.write("None")

        st.subheader("Overall Recommendation")
        if similarity_score < 60:
            st.error("Low match – Focus on learning missing skills and updating your resume keywords.")
        elif similarity_score < 80:
            st.warning("Medium match – You are close, highlight relevant project experience.")
        else:
            st.success("High match – Your profile is strong for this role!")