import streamlit as st
from PIL import Image
import pandas as pd
import base64

# Set page configuration
st.set_page_config(
    page_title="Your Name | Professional Portfolio",
    page_icon="👨‍💼",
    layout="wide"
)

# Custom CSS styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")  # You can create a separate style.css file or define styles below

# Sidebar
with st.sidebar:
    st.title("Navigation")
    menu = ["Home", "About Me", "Projects", "Skills", "Experience", "Education", "Contact"]
    choice = st.radio("Go to", menu)
    
    # Contact information in sidebar
    st.markdown("---")
    st.subheader("Contact Info")
    st.write("📧 email@example.com")
    st.write("📞 +1 (123) 456-7890")
    st.write("📍 Your City, Country")
    
    # Social media links
    st.markdown("---")
    st.subheader("Connect With Me")
    st.markdown("[LinkedIn](#)", unsafe_allow_html=True)
    st.markdown("[GitHub](#)", unsafe_allow_html=True)
    st.markdown("[Twitter](#)", unsafe_allow_html=True)

# Profile image (replace with your own image)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        background-attachment: fixed;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

#set_png_as_page_bg('background.png')  # Uncomment if you want a background image

# Home Page
if choice == "Home":
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Replace with your profile image
        profile_img = Image.open("profile.jpg")  # Place your image in the same folder
        st.image(profile_img, width=250)
    
    with col2:
        st.title("Your Name")
        st.subheader("Your Profession/Tagline")
        st.write("""
        Welcome to my professional portfolio! I'm passionate about [your field] 
        with expertise in [key skills]. This portfolio showcases my work, skills, 
        and professional journey.
        """)
        
        # Download resume button
        with open("resume.pdf", "rb") as pdf_file:  # Replace with your resume file
            PDFbyte = pdf_file.read()
        
        st.download_button(
            label="Download Resume",
            data=PDFbyte,
            file_name="YourName_Resume.pdf",
            mime="application/octet-stream",
        )

# About Me Page
elif choice == "About Me":
    st.title("About Me")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("about_me.jpg", width=300)  # Optional about me image
    
    with col2:
        st.write("""
        ## Who Am I?
        A detailed introduction about yourself, your professional journey, 
        and what drives you. Highlight your values, approach to work, and 
        what makes you unique.
        
        ### Professional Philosophy
        - Value 1: Explanation
        - Value 2: Explanation
        - Value 3: Explanation
        
        ### Personal Interests
        Outside of work, I enjoy [hobby 1], [hobby 2], and [hobby 3]. 
        These activities help me [benefit they provide].
        """)

# Projects Page
elif choice == "Projects":
    st.title("My Projects")
    st.markdown("---")
    
    # Project 1
    with st.expander("**Project 1: Project Name**", expanded=True):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image("project1.jpg", width=250)  # Project image
        
        with col2:
            st.write("""
            ### Overview
            Brief description of the project, its purpose, and your role.
            
            ### Technologies Used
            - Technology 1
            - Technology 2
            - Technology 3
            
            ### Key Achievements
            - Achievement 1
            - Achievement 2
            
            [View Project](#) | [GitHub Repository](#)
            """)
    
    # Project 2
    with st.expander("**Project 2: Project Name**", expanded=False):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image("project2.jpg", width=250)  # Project image
        
        with col2:
            st.write("""
            ### Overview
            Brief description of the project, its purpose, and your role.
            
            ### Technologies Used
            - Technology 1
            - Technology 2
            - Technology 3
            
            ### Key Achievements
            - Achievement 1
            - Achievement 2
            
            [View Project](#) | [GitHub Repository](#)
            """)
    
    # Add more projects as needed

# Skills Page
elif choice == "Skills":
    st.title("My Skills")
    st.markdown("---")
    
    # Technical Skills
    st.subheader("Technical Skills")
    tech_skills = {
        "Skill Category 1": ["Skill 1", "Skill 2", "Skill 3"],
        "Skill Category 2": ["Skill 1", "Skill 2", "Skill 3"],
        "Skill Category 3": ["Skill 1", "Skill 2", "Skill 3"],
    }
    
    for category, skills in tech_skills.items():
        with st.expander(f"**{category}**"):
            for skill in skills:
                st.write(f"- {skill}")
    
    # Professional Skills
    st.subheader("Professional Skills")
    prof_skills = ["Communication", "Team Leadership", "Project Management", "Problem Solving"]
    for skill in prof_skills:
        st.write(f"- {skill}")

# Experience Page
elif choice == "Experience":
    st.title("Professional Experience")
    st.markdown("---")
    
    # Job 1
    with st.expander("**Company 1 | Job Title** (Year - Year)", expanded=True):
        st.write("""
        ### Responsibilities
        - Responsibility 1
        - Responsibility 2
        - Responsibility 3
        
        ### Achievements
        - Achievement 1
        - Achievement 2
        """)
    
    # Job 2
    with st.expander("**Company 2 | Job Title** (Year - Year)", expanded=False):
        st.write("""
        ### Responsibilities
        - Responsibility 1
        - Responsibility 2
        - Responsibility 3
        
        ### Achievements
        - Achievement 1
        - Achievement 2
        """)

# Education Page
elif choice == "Education":
    st.title("Education & Certifications")
    st.markdown("---")
    
    # Education 1
    with st.expander("**Degree Name** - University Name (Year - Year)", expanded=True):
        st.write("""
        - Major: Your Major
        - GPA: X.XX/4.00
        - Relevant Coursework: Course 1, Course 2, Course 3
        """)
    
    # Certifications
    st.subheader("Certifications")
    certs = {
        "Certification 1": "Issuing Organization (Year)",
        "Certification 2": "Issuing Organization (Year)",
    }
    for cert, details in certs.items():
        st.write(f"- **{cert}**: {details}")

# Contact Page
elif choice == "Contact":
    st.title("Get In Touch")
    st.markdown("---")
    
    contact_form = """
    <form action="https://formsubmit.co/your@email.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here"></textarea>
        <button type="submit">Send</button>
    </form>
    """
    
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>© 2023 Your Name. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
