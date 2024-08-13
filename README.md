# Chronic-Kidney-Disease-Notebook-from-Basic-to-Advance-of-ML
In this notebook i have performed Basics to Advanced Machine Learning Algorithms for better to understand the model accuracy at various level's.  . Finally CKD dataset has yielded an impressive accuracy of 98% across multiple algorithms. 

# Aim of the CKD Dataset
The aim of the Chronic Kidney Disease (CKD) dataset is to facilitate the study and prediction of chronic kidney disease by using machine learning and statistical analysis techniques. This dataset typically contains medical and laboratory information about patients, including features such as age, blood pressure, specific laboratory tests (e.g., serum creatinine, blood urea, hemoglobin), and other relevant clinical data.

Key objectives include:

* Predicting Chronic Kidney Disease: Developing predictive models to accurately classify whether a patient has CKD based on the available features.
* Understanding Risk Factors: Identifying key factors or indicators that contribute to the onset and progression of CKD, which can aid in early detection and intervention.
* Improving Clinical Decision-Making: Assisting healthcare professionals in making informed decisions regarding diagnosis, treatment, and management of CKD.
* This dataset is widely used for research, education, and the development of machine learning models aimed at improving patient outcomes in the context of CKD.

# Understanding about the CKD Dataset Features

<table border="1" cellpadding="5" cellspacing="0">
  <thead>
    <tr>
      <th>Feature</th>
      <th>Description</th>
      <th>Type</th>
      <th>Example</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Age</td>
      <td>The age of the patient in years.</td>
      <td>Numerical</td>
      <td>48 years</td>
    </tr>
    <tr>
      <td>Blood Pressure (bp)</td>
      <td>The patient's blood pressure, typically measured in mm Hg.</td>
      <td>Numerical</td>
      <td>80 (diastolic pressure)</td>
    </tr>
    <tr>
      <td>Specific Gravity (sg)</td>
      <td>Measure of urine concentration, indicating kidney's ability to concentrate urine.</td>
      <td>Categorical (1.005-1.025)</td>
      <td>1.020</td>
    </tr>
    <tr>
      <td>Albumin (al)</td>
      <td>Presence of albumin in urine, indicating kidney damage.</td>
      <td>Categorical (0-5)</td>
      <td>2</td>
    </tr>
    <tr>
      <td>Sugar (su)</td>
      <td>Presence of sugar in urine, possibly indicating diabetes or kidney disease.</td>
      <td>Categorical (0-5)</td>
      <td>3</td>
    </tr>
    <tr>
      <td>Red Blood Cells (rbc)</td>
      <td>Presence of red blood cells in urine, abnormal levels may indicate kidney disease.</td>
      <td>Categorical (normal, abnormal)</td>
      <td>abnormal</td>
    </tr>
    <tr>
      <td>Pus Cell (pc)</td>
      <td>Presence of pus cells in urine, indicating infection or inflammation in kidneys.</td>
      <td>Categorical (normal, abnormal)</td>
      <td>normal</td>
    </tr>
    <tr>
      <td>Pus Cell Clumps (pcc)</td>
      <td>Clumps of pus cells in urine, suggesting a more severe infection.</td>
      <td>Categorical (present, not present)</td>
      <td>not present</td>
    </tr>
    <tr>
      <td>Bacteria (ba)</td>
      <td>Presence of bacteria in urine, possibly indicating a urinary tract infection.</td>
      <td>Categorical (present, not present)</td>
      <td>present</td>
    </tr>
    <tr>
      <td>Blood Glucose Random (bgr)</td>
      <td>Random measurement of blood glucose levels, indicating blood sugar control.</td>
      <td>Numerical (mg/dL)</td>
      <td>121 mg/dL</td>
    </tr>
    <tr>
      <td>Blood Urea (bu)</td>
      <td>Level of urea in blood, an indicator of kidney function.</td>
      <td>Numerical (mg/dL)</td>
      <td>36 mg/dL</td>
    </tr>
    <tr>
      <td>Serum Creatinine (sc)</td>
      <td>Level of creatinine in blood, a marker of kidney function.</td>
      <td>Numerical (mg/dL)</td>
      <td>1.2 mg/dL</td>
    </tr>
    <tr>
      <td>Sodium (sod)</td>
      <td>Level of sodium in blood, affecting kidney function.</td>
      <td>Numerical (mEq/L)</td>
      <td>140 mEq/L</td>
    </tr>
    <tr>
      <td>Potassium (pot)</td>
      <td>Level of potassium in blood, important for kidney function.</td>
      <td>Numerical (mEq/L)</td>
      <td>4.5 mEq/L</td>
    </tr>
    <tr>
      <td>Hemoglobin (hemo)</td>
      <td>Amount of hemoglobin in blood, can be low in kidney disease.</td>
      <td>Numerical (g/dL)</td>
      <td>13.5 g/dL</td>
    </tr>
    <tr>
      <td>Packed Cell Volume (pcv)</td>
      <td>Volume percentage of red blood cells in blood, related to kidney function.</td>
      <td>Numerical (Percentage)</td>
      <td>41%</td>
    </tr>
    <tr>
      <td>White Blood Cell Count (wc)</td>
      <td>Number of white blood cells in blood, indicating infection or inflammation.</td>
      <td>Numerical (cells/cumm)</td>
      <td>8400 cells/cumm</td>
    </tr>
    <tr>
      <td>Red Blood Cell Count (rc)</td>
      <td>Number of red blood cells in blood, related to overall blood health.</td>
      <td>Numerical (millions/cumm)</td>
      <td>5.2 millions/cumm</td>
    </tr>
    <tr>
      <td>Hypertension (htn)</td>
      <td>Indicates whether the patient has hypertension, often associated with CKD.</td>
      <td>Categorical (yes, no)</td>
      <td>yes</td>
    </tr>
    <tr>
      <td>Diabetes Mellitus (dm)</td>
      <td>Indicates whether the patient has diabetes mellitus, which is a risk factor for CKD.</td>
      <td>Categorical (yes, no)</td>
      <td>yes</td>
    </tr>
    <tr>
      <td>Coronary Artery Disease (cad)</td>
      <td>Indicates whether the patient has coronary artery disease, another risk factor for CKD.</td>
      <td>Categorical (yes, no)</td>
      <td>no</td>
    </tr>
    <tr>
      <td>Appetite</td>
      <td>Indicates the patient's appetite, which can be affected by CKD.</td>
      <td>Categorical (good, poor)</td>
      <td>good</td>
    </tr>
    <tr>
      <td>Pedal Edema</td>
      <td>Indicates whether the patient has swelling in the lower extremities, a common symptom in CKD.</td>
      <td>Categorical (yes, no)</td>
      <td>no</td>
    </tr>
    <tr>
      <td>Anemia</td>
      <td>Indicates whether the patient has anemia, which can be a complication of CKD.</td>
      <td>Categorical (yes, no)</td>
      <td>no</td>
    </tr>
    <tr>
      <td>Class</td>
      <td>Indicates whether the patient has CKD or not (the target variable).</td>
      <td>Categorical (ckd, notckd)</td>
      <td>ckd</td>
    </tr>
  </tbody>
</table>


![newplot](https://github.com/user-attachments/assets/ca1181c2-20a2-4246-abde-0096b2f18854)


# Conclusion:

* The analysis of the Chronic Kidney Disease (CKD) dataset has yielded an impressive accuracy of 98% across multiple algorithms. This high accuracy indicates that the models are highly effective in predicting the presence or absence of CKD based on the available data.
* The 98% accuracy suggests that the models are performing exceptionally well, correctly classifying CKD cases and non-cases in the majority of instances.
