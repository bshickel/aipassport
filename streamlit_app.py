import altair as alt
import pandas as pd
import streamlit as st
import time
from google import genai

st.set_page_config(page_title="AI Passport Notebook Test", page_icon="📘")
st.title("📘 AI Passport Notebook Test")
st.caption('Demo of Streamlit functionality for AI Passport notebooks.')

@st.cache_data
def load_data():
    df = pd.read_csv("data/eicu_demo.csv")
    return df

df = load_data()

st.markdown(
    """
    ### 🗂️ Data-Centric Artificial Intelligence
    Data is the fuel that powers artificial intelligence (AI), or more precisely, machine learning (ML) - the field of AI focused on statistical algorithms and techniques that can learn useful patterns from data without being explicitly programmed. Without volumes of machine-readable data, AI systems would not be as advanced as they are today, and may still be limited to the fixed rule-based AI systems from the 1950's and 60's.

    While algorithms tend to garner the most publicity, in many (or even most) cases it is improvements in the data being used to train these algorithms that often leads to better and more useful AI models. This is especially true for AI using real-world patient data, such as clinical datasets derived from electronic health record (EHR) systems, which can contain errors, outliers, missing information, and other irregularities resulting from the complexities of real-world clinical care. 

    Source data undergoes several steps before being used to train a machine learning model, (e.g., acquisition, exploration, analysis, transformation, preprocessing, feature engineering, structuring for specific algorithms, and more). It is often these steps that require the most time and effort on an AI project. Data-centric AI is a recent school of thought which advocates for standardized practices which prioritize data curation over algorithm tuning.

    As the landscape of AI algorithms and models continues to mature, systematic clinical data engineering and quality assurance will likely yield higher-quality models that drive better patient and health system outcomes. It is essential for clinical AI researchers to possess the skills, tools, and domain knowledge necessary for a comprehensive understanding of the data used to build healthcare AI systems.

    Before diving into an AI project, taking the time to become intimately familiar with the data you will be modeling can save you time, energy, and headache later if things aren't turning out as you initially expected. Exploratory data analysis can help you identify important data quality issues early in the AI process that can save you countless hours in the future.")
""")

with st.container(border=True):
    
    st.markdown('### Question 1')
    r1 = st.radio(label='**Should you explore your data before training models?**',
                options=['Yes', 'No'],
                index=None)
    if r1 == 'Yes':
        # if r1_balloons is not True:
        #     st.balloons()
        #     r1_balloons = True

        st.success('**🎉 Correct!**')

        st.markdown( 
            """
            **Exploratory data analysis has many benefits:**
            - It can help identify major problems with the dataset that may challenge your assumptions and/or require returning to the data source or rethinking your entire question and/or approach.
            - You can quickly identify data quality issues such as missing values, outliers, and otherwise incomplete, irrelevant, or misaligned data elements.
            - A hands-on understanding of the data can influence your downstream selection of the best technical approach.
            - A better understanding of your population is likely to lead to a more accurate interpretation of your results.
            - Spending time early can help minimizing the amount of backtracking and repetition as issues are discovered later in pipeline.
            - Furthermore, for deployed systems that rely on continuous integration (CI), deployment (CD), and training (CT) on new data over time, continuous iterations of rigorous data analysis can help spot issues such as data drift (where models trained on older data become less relevant for newer data streams)
        """)

    elif r1 == 'No':
        st.error('Try again.')
            
st.markdown("""
The most straightforward type of dataset used for machine learning is **structured** (or *tabular*). A structured dataset can be thought of as a two-dimensional table or spreadsheet. In other words, the rows and columns have clearly defined meaning.
""")


st.markdown("""
    In this tutorial, we will use a real-world dataset of 1,841 patients who underwent 2,174 total inpatient hospitalizations across 186 different hospitals spread throughout the United States. This is a subset of the [eICU Collaborative Research Database](https://eicu-crd.mit.edu/), a publicly available database which contains over 200,000 patient stays.

    Here's a quick summary of the data we'll be loading from eICU:

    * **29 model inputs, a.k.a. features (X)**. These are the patient characteristics that we'll pass into the model in order to predict the outcome.
        * Demographic variables (e.g., `age`, `gender`, `ethnicity`)
        * Admission information (e.g., admission_weight, admission source)
        * Hospital information (e.g., geographic location, number of beds, whether it's a teaching hospital)
        * 18 laboratory test results taken during their hospitalization 
            
    * **1 model output, a.k.a. target/outcome (Y)**. This variable is what the model will be trained to predict based on the inputs above.
        * **In-hospital mortality**, i.e., whether the patient died in the hospital or was discharged successfully. This is a binary outcome which is recorded as either 0 (survival) or 1 (death) for each patient.
        * This dataset contains other columns that could serve as interesting outcomes to predict (e.g., `discharge_location` or `weight_discharge`). However, we leave these as a supplemental exercise for learners, and in this notebook, we focus exclusively on predicting `in_hospital_mortality`.
        """)

with st.container(border=True):
    st.markdown('### Question 2')
    st.markdown('**Which variable types does this dataset contain?** (*Select all that apply.*)')
    c1 = st.checkbox('Patient demographic information')
    c2 = st.checkbox('Laboratory test results')
    c3 = st.checkbox('Geographic hospital location')
    c4 = st.checkbox('Patient Social Security Number (SSN)')

    if (not c4) and (c1 and c2 and c3):
        st.success('**🎉 Correct!**')

    if c4:
        st.error('**❌ This dataset does not contain protected health information (PHI) as defined by the [HIPAA Privacy Rule](https://www.hhs.gov/hipaa/for-professionals/privacy/laws-regulations/index.html)**')

st.markdown('\n')
st.markdown('\n')
with st.container(border=True):
    st.title('🧭 Data Explorer')

    hospital_region = st.pills(
        label="Hospital Region",
        options=["Midwest", "South", "West", "Northeast"],
        selection_mode="multi",
        default="South"

    )

    age = st.slider(label="Patient Age",
                    min_value=15,
                    max_value=90,
                    value=(18, 70))

    height = st.slider(label="Height (cm)",
                    min_value=0,
                    max_value=600,
                    value=(162, 178))

    weight = st.slider(label="Weight (kg)",
                    min_value=0,
                    max_value=300,
                    value=(65, 96))

    gender = st.pills(
        label="Gender",
        options=["Male", "Female"],
        selection_mode="multi",
        default="Female"
    )

    # Filter the dataframe based on the widget input and reshape it.
    df_filtered = df[(df["hospital_region"].isin(hospital_region))
                    & (df["age"].between(age[0], age[1]))
                    & (df["height"].between(height[0], height[1]))
                    & (df["weight"].between(weight[0], weight[1]))
                    & (df["gender"].isin(gender))]

    st.dataframe(
        df_filtered,
        use_container_width=True,
        # column_config={"year": st.column_config.TextColumn("Year")},
    )

    labs = ['lab_bun', 'lab_creatinine', 'lab_sodium', 'lab_hct', 'lab_wbc', 'lab_glucose',
            'lab_potassium', 'lab_hgb', 'lab_chloride', 'lab_platelets', 'lab_rbc', 'lab_calcium',
            'lab_mcv', 'lab_mchc', 'lab_bicarbonate', 'lab_mch', 'lab_rdw', 'lab_albumin']

    df_grouped = df_filtered.groupby('in_hospital_mortality')[
        labs].mean().reset_index()
    df_melted = df_grouped.melt(id_vars="in_hospital_mortality",
                                var_name="Lab Test",
                                value_name="Mean Value")
    df_melted["in_hospital_mortality"] = df_melted["in_hospital_mortality"].map(
        {0: "In-Hospital Mortality", 1: "Survival"})


    chart = alt.Chart(df_melted).mark_bar().encode(
        y=alt.X("Lab Test:N", title="Lab Test"),
        x=alt.Y("Mean Value:Q", title="Mean Lab Value"),
        color=alt.Color("in_hospital_mortality:N", title="Outcome"),

        tooltip=["Lab Test", "Mean Value", "in_hospital_mortality"]
    ).properties(
        width=150,
        height=300,
        title="Mean Lab Test Values by Outcome"
    )

    st.altair_chart(chart, use_container_width=True)

st.markdown("""
            Once you determine your what you'd like your model to predict, it is important to identify any variables that may pose a risk of **data leakage**, or information being available to your model during training that would not typically be available when making predictions in the future, and exclude them from input features used to train the model.
            
            For example, if our goal is to predict `in_hospital_mortality`, we cannot include `discharge_location` as an input feature, since one of the possible values of `discharge_location` is `Death`. If we knew the patient's discharge location, we would already know if they survived, and a mortality-prediction model would serve no purpose. If we include such features in our training, we would see extremely good model performance, but those results would be completely misleading and not very useful.
    """)

with st.container(border=True):
    st.markdown("# 🏋️‍♂️ Model Trainer")

    st.multiselect(
        label="**Select input features** (*What will your model use to make predictions?*)",
        options=df.columns,
        placeholder='Choose one or more input variables')

    st.selectbox(label="**Select prediction target** (*What will your model be predicting?*)",
                options=[c for c in df.columns if c not in ['patient_id', 'admission_id', 'hospital_id']],
                placeholder='Choose an outcome variable',
                index=None)


    model = st.pills(
        label="**Select model** (*What algorithm will you use to map inputs to outputs?*)",
        options=["Logistic Regression",
                "Support Vector Machine (SVM)",
                "Random Forest",
                "XGBoost",
                "Recurrent Neural Network (RNN)",
                "Convolutional Neural Network (CNN)",
                "Transformer"],
        selection_mode="single",
        default="Logistic Regression"
    )

    if model == "Logistic Regression":
        model_text = "A simple yet powerful classification algorithm that models the probability of an outcome using a logistic function. It is commonly used for binary classification problems, such as predicting disease presence (yes/no).\n\nWorks best with structured, tabular data where relationships between features and the binary outcome can be well approximated by a linear decision boundary."
    elif model == "Support Vector Machine (SVM)":
        model_text = "A supervised learning algorithm that finds the optimal hyperplane to separate data into different classes. It works well in high-dimensional spaces and is effective for both linear and non-linear classification using kernel tricks.\n\nPerforms well on high-dimensional structured data, especially when the classes are separable or can be effectively mapped into a higher-dimensional space using kernel tricks."
    elif model == "Random Forest":
        model_text = "An ensemble learning method that builds multiple decision trees and averages their outputs to improve accuracy and reduce overfitting. It is widely used for structured data problems, such as risk prediction in healthcare.\n\nExcels with structured tabular data, especially when there are many categorical and numerical features with potential non-linear relationships."
    elif model == "XGBoost":
        model_text = "A high-performance boosting algorithm that builds trees sequentially, optimizing performance with techniques like regularization and weighted voting. It is highly efficient and commonly used in competitive machine learning.\n\nWorks best with structured tabular data, particularly in cases where feature interactions and non-linearity play a critical role in prediction."
    elif model == "Recurrent Neural Network (RNN)":
        model_text = "A neural network designed for sequential data, such as time series or natural language. It has connections that allow information to persist across steps, making it suitable for applications like patient health monitoring and speech recognition.\n\nSuited for sequential data, such as time series, speech, and clinical records where past information influences future predictions."
    elif model == "Convolutional Neural Network (CNN)":
        model_text = "A deep learning architecture specialized for processing structured grid-like data, such as images. It uses convolutional layers to extract spatial features, making it a go-to model for medical imaging and computer vision tasks.\n\nBest for image, video, and spatial data where local patterns and hierarchical feature extraction are crucial."
    elif model == "Transformer":
        model_text = "A neural network architecture designed for handling sequential data efficiently, using self-attention mechanisms to process entire sequences in parallel. Transformers power state-of-the-art language models, such as GPT and BERT, and have applications in clinical text analysis and genomics.\n\nIdeal for sequential data like text, genomics, and patient records, where long-range dependencies and contextual understanding are important."
    else:
        model_text = ''

    model_string = str(model) if model is not None else '*Select a model above.*'
    st.info('**' + model_string + '** \n\n' + '' + model_text, icon='⚙️')

    st.markdown('**Model Parameters**')
    with st.container(border=True):
        p1 = st.pills(
            label="**Parameter XYZ**",
            options=["Option 1", "Option 2", "Option 3"],
            selection_mode="single")
        st.caption("In a " + model + " model, **XYZ** controls the ...")

        st.markdown('') 
    
        p2 = st.slider(label="**Parameter XYZ**",
                       min_value=0.0,
                       max_value=1.0,
                       step=0.01)
        st.caption("In a " + model + " model, **XYZ** controls the ...")

        st.markdown('') 

        p2 = st.slider(label="**Parameter XYZ**",
                       key=11,
                       min_value=0.0,
                       max_value=1.0,
                       step=0.01)
        st.caption("In a " + model + " model, **XYZ** controls the ...")

        st.markdown('') 

    st.markdown('**Training Options**')
    with st.container(border=True):
        p3 = st.toggle('Use **XYZ**', key=0)
        st.caption("In a " + model + " model, using **XYZ** will...")

        st.markdown('') 

        p4 = st.toggle('Use **XYZ**', key=1) 
        st.caption("In a " + model + " model, using **XYZ** will...")

        st.markdown('') 

        p5 = st.toggle('Use **XYZ**', key=2) 
        st.caption("In a " + model + " model, using **XYZ** will...")

    def progress_bar(text="Operation in progress. Please wait."):
        bar = st.progress(0, text=text)
        for percent_complete in range(100):
            time.sleep(0.01)
            bar.progress(percent_complete + 1, text=text)

    if st.button(label="Train " + model + ' model', type='primary', use_container_width=True):
        progress_bar()
        st.success('🎉 Model training complete!')

st.markdown('\n')
st.markdown('\n')

with st.container(border=True):
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    st.markdown('# 🤖 Chat')
    st.caption('Gemini 2.0 Flash')
    
    history = st.container(height=200)
    with history:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("What's on your mind?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with history:
            with st.chat_message("user"):
                st.markdown(prompt)

        contents = [
            {'role': message['role'],
             'parts': [{'text': message['content']}]}
             for message in st.session_state.messages
        ]

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=contents
        )
        st.session_state.messages.append({"role": "assistant", "content": response.text})

        with history:
            with st.chat_message("assistant"):
                st.markdown(response.text)
