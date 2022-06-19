import streamlit as st
import pickle
import numpy as np
import pandas as pd

#defining methods to load pipelines
def load_model1():
    with open('saved_steps1.pkl', 'rb') as file1:
        data1 = pickle.load(file1)
    return data1

def load_model2():
    with open('saved_steps2.pkl', 'rb') as file2:
        data2 = pickle.load(file2)
    return data2

def load_model3():
    with open('saved_steps3.pkl', 'rb') as file3:
        data3 = pickle.load(file3)
    return data3

#fetching pipeline elements
datax1 = load_model1()
datax2 = load_model2()
datax3 = load_model3()

#storing elements of each pipeline
mod1 = datax1["model1"]
oe1 = datax1["oe"]
target_le = datax1["target"]
fs1 = datax1["feature_sel"]

mod2 = datax2["model2"]
oe1p = datax2["oep"]
fs2 = datax2["feature_sele"]

mod3 = datax3["model3"]

#defining method to show predict page
def show_predict_page():
    st.title("IT Career Prediction System")


    st.warning(
        "Please enter the data below. All fields are required. Select N/A if you do not have the skill/experience.")

    criteria1 = ("Poor",
                 "Moderate",
                 "Good",
                 "Excellent",
                 "N/A")

    criteria2 = ("Beginner",
                 "Intermediate",
                 "Expert",
                 "N/A")

    criteria3 = ("Poor",
                 "Moderate",
                 "Good",
                 "Excellent")

    st.title("Skills/Experiences")
    coding = st.selectbox("Coding skill", criteria1)
    ui_ux = st.selectbox('UI/UX Technologies', criteria2)
    unit_test = st.selectbox("Unit testing and test automation", criteria2)
    soft_eng_con = st.selectbox("Software Engineering Concepts(Data structures, algorithms etc.)", criteria2)
    problem = st.selectbox('Problem solving skills', criteria1)
    soft_deb = st.selectbox("Software Debugging", criteria2)
    db = st.selectbox("Database skills", criteria2)
    soft_perf = st.selectbox('Software performance optimization', criteria2)
    analytical = st.selectbox("Analytical skills", criteria1)
    version_con = st.selectbox("Version controlling (Github, Bitbucket etc.)", criteria2)
    soft_arch = st.selectbox('Software architectural and design patterns', criteria2)
    fam_linux = st.selectbox("Familiarity with Linux", criteria1)
    server_admin = st.selectbox("Server Administration (Linux/Windows)", criteria2)
    com_net = st.selectbox('Computer Networking', criteria2)
    math = st.selectbox("Mathematics", criteria2)
    stat = st.selectbox('Statistics', criteria2)
    cloud = st.selectbox("Cloud technologies (AWS, Azure etc.)", criteria2)
    data_vis = st.selectbox("Data visualization (PowerBI etc.)", criteria2)
    ml = st.selectbox("Machine learning skills", criteria2)
    deep = st.selectbox("Deep learning skills", criteria2)
    nlp = st.selectbox('Natural Language Processing', criteria2)
    iot = st.selectbox('Internet of Things(IoT) knowledge', criteria2)
    office = st.selectbox('Microsoft Office skills', criteria2)
    ci_cd = st.selectbox('CI/CD tools and technologies', criteria2)
    soft_dev_meth = st.selectbox("Software development methodologies (Agile etc.)", criteria2)
    web_app_ser = st.selectbox('Web and application servers (Tomcat, Weblogic etc.)', criteria2)
    web_ser = st.selectbox('Web services (RESTful Web Services etc.)', criteria2)
    micro = st.selectbox('Microservices', criteria2)
    cms = st.selectbox("CMS platforms (Joomla, Wordpress etc.)", criteria2)
    comm_skill = st.selectbox('Communication skills (English written and verbal)', criteria1)
    pres_skills = st.selectbox('Presentation skills', criteria1)
    tech_wr = st.selectbox("Technical writing / documentation skills", criteria1)
    oop = st.selectbox('Object Oriented Programming Concepts and Design Patterns', criteria2)
    fam_pro = st.selectbox('Familiarity with prototype tools (Invision, Axure etc.)', criteria2)
    big_data = st.selectbox("Big data tools (Hadoop, Spark etc.)", criteria2)
    data_ware = st.selectbox("Data Warehouse Solutions (Redshift, Snowflakes etc.)", criteria2)
    req_modl = st.selectbox('Requirement modeling techniques/tools', criteria2)
    cus_ser = st.selectbox('Customer service skills', criteria1)
    gen_com = st.selectbox('General Computer skills', criteria1)
    inter_skill = st.selectbox('Interpersonal skills', criteria1)
    time_mng = st.selectbox('Time management skills', criteria1)
    multi_task = st.selectbox("Multitasking skills", criteria1)
    ar_vr = st.selectbox('Augmented Reality / Virtual Reality', criteria2)

    st.title("Personality")
    tp = st.selectbox('Team player', criteria3)
    ability_press = st.selectbox("Ability to work under pressure", criteria3)
    positive = st.selectbox("Positive outlook", criteria3)
    innov = st.selectbox('Innovative mindset', criteria3)
    pass_new = st.selectbox('Passion of new technologies', criteria3)
    att_learn = st.selectbox('Attitude and desire to learn', criteria3)
    adapt_dyn = st.selectbox("Adaptability to a dynamic environment", criteria3)
    ability_res = st.selectbox("Ability to take responsibility", criteria3)
    ability_indiv = st.selectbox("Ability to work as an individual", criteria3)

    ok = st.button("Click to see career")

    if ok:
        X1 = np.array([[coding, ui_ux, unit_test, soft_eng_con, problem, soft_deb, db, soft_perf, analytical, version_con, soft_arch, fam_linux, server_admin,
                        com_net, math, stat, cloud, data_vis, ml, deep, nlp, iot, office, ci_cd, soft_dev_meth, web_app_ser, web_ser, micro, cms,
                        comm_skill, pres_skills, tech_wr, oop, fam_pro, big_data, data_ware, req_modl, cus_ser, gen_com, inter_skill, time_mng, multi_task, ar_vr]])

        X1 = oe1.transform(X1)
        X1_df = pd.DataFrame(X1)
        X1_df = X1_df.drop(columns=[30])
        X1_arr = np.array(X1_df)
        X1_out = fs1.transform(X1_arr)
        X1_out = mod1.predict(X1_out)
        x1_out = pd.DataFrame(X1_out, columns=['c1'])


        X2 = np.array([[tp, ability_press, positive, innov, pass_new, att_learn, adapt_dyn, ability_res, ability_indiv]])
        X2 = oe1p.transform(X2)
        X2_out = fs2.transform(X2)
        X2_out = mod2.predict(X2_out)
        x2_out = pd.DataFrame(X2_out, columns=['c2'])


        c_df= x1_out.join(x2_out)


        final = mod3.predict(c_df)

        career_name = target_le.inverse_transform(final)

        st.title("You may select : " + career_name[0])



