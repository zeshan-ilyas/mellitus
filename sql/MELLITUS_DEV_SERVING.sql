----CREATING VIEWS 

  

-- View showing AVG AGE 

CREATE OR REPLACE VIEW AVG_AGE_VIEW AS 

SELECT  

    AVG(AGE) AS AVG_AGE 

FROM MELLITUS_DEV_CURATED.PUBLIC.PATIENT_DIM ; 

  

  

SELECT * FROM AVG_AGE_VIEW; 

  

--View Showing AVG BMI 

CREATE OR REPLACE VIEW AVG_BMI_VIEW AS 

SELECT  

    AVG(BMI) AS AVG_BMI 

FROM MELLITUS_DEV_CURATED.PUBLIC.HEALTH_METRICS_DIM ; 

  

-- View Showing Gender Distribution 

CREATE OR REPLACE VIEW GENDER_DISTRIBUTION_VIEW AS 

SELECT  

    GENDER, 

    COUNT(*) AS PATIENT_COUNT 

FROM MELLITUS_DEV_CURATED.PUBLIC.PATIENT_DIM 

GROUP BY GENDER; 

  

  

--Validate the Gender View 

SELECT * FROM GENDER_DISTRIBUTION_VIEW; 

  

--View showing Hypertension Percentage 

CREATE OR REPLACE VIEW HYPERTENSION_PERCENTAGE_VIEW AS 

SELECT  

    ROUND((SUM(CASE WHEN HYPERTENSION = 1 THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2) AS PERCENT_HYPERTENSION 

FROM MELLITUS_DEV_CURATED.PUBLIC.HEALTH_METRICS_DIM; 

  

  

-- View Showing Diabetes Percentage 

CREATE OR REPLACE VIEW DIABETES_PERCENTAGE_VIEW AS 

SELECT  

    ROUND((SUM(CASE WHEN DIABETES = 1 THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2) AS PERCENT_DIABETES 

FROM MELLITUS_DEV_CURATED.PUBLIC.DIABETES_FACT; 

  

  

--View showing Fanily Diabetes History 

CREATE OR REPLACE VIEW FAMILY_DIABETES_HISTORY_VIEW AS 

SELECT  

    FAMILY_DIABETES_HISTORY, 

    COUNT(*) AS PATIENT_COUNT 

FROM MELLITUS_DEV_CURATED.PUBLIC.HEALTH_METRICS_DIM 

GROUP BY FAMILY_DIABETES_HISTORY; 

  

--View Showing Diet Type Count 

CREATE OR REPLACE VIEW DIET_TYPE_COUNT_VIEW AS 

SELECT  

    DIET_TYPE, 

    COUNT(*) AS PATIENT_COUNT 

FROM MELLITUS_DEV_CURATED.PUBLIC.LIFESTYLE_DIM 

GROUP BY DIET_TYPE; 

  

--PHYSICAL ACTIVITY VIEW 

CREATE OR REPLACE VIEW PHYSICAL_ACTIVITY_COUNT_VIEW AS 

SELECT  

    PHYSICAL_ACTIVITY_LEVEL, 

    COUNT(*) AS PATIENT_COUNT 

FROM MELLITUS_DEV_CURATED.PUBLIC.LIFESTYLE_DIM 

GROUP BY PHYSICAL_ACTIVITY_LEVEL; 

  

--AVG SLEEP DURATION VIEW 

CREATE OR REPLACE VIEW AVG_SLEEP_DURATION_VIEW AS 

SELECT  

    AVG(SLEEP_DURATION) AS AVG_SLEEP_DURATION 

FROM MELLITUS_DEV_CURATED.PUBLIC.LIFESTYLE_DIM; 

  

  

--Viw=ew Counting Stress Level 

CREATE OR REPLACE VIEW STRESS_LEVEL_COUNT_VIEW AS 

SELECT  

    STRESS_LEVEL, 

    COUNT(*) AS PATIENT_COUNT 

FROM MELLITUS_DEV_CURATED.PUBLIC.LIFESTYLE_DIM 

GROUP BY STRESS_LEVEL; 

  

--View Showing Alcihol Consumption 

CREATE OR REPLACE VIEW ALCOHOL_CONSUMPTION_COUNT_VIEW AS 

SELECT  

    ALCOHOL_CONSUMPTION, 

    COUNT(*) AS PATIENT_COUNT 

FROM MELLITUS_DEV_CURATED.PUBLIC.LIFESTYLE_DIM 

GROUP BY ALCOHOL_CONSUMPTION; 

  

--View for Patients with Hypertension and Diabetes 

CREATE OR REPLACE VIEW HYDRATED_DIABETES_PATIENTS AS 

SELECT DISTINCT  

    p.PATIENT_ID, 

    p.GENDER, 

    p.AGE, 

    m.HYPERTENSION, 

    d.DIABETES, 

    m.DIABETES_PEDIGREE_FUNCTION, 

    m.FAMILY_DIABETES_HISTORY, 

    m.BMI, 

    m.WEIGHT, 

    p.AGE_GROUP 

FROM MELLITUS_DEV_CURATED.PUBLIC.PATIENT_DIM p 

JOIN MELLITUS_DEV_CURATED.PUBLIC.HEALTH_METRICS_DIM m  

    ON p.PATIENT_ID = m.PATIENT_ID 

JOIN MELLITUS_DEV_CURATED.PUBLIC.DIABETES_FACT d 

    ON p.PATIENT_ID = d.PATIENT_ID 

WHERE m.HYPERTENSION = 1  

  AND d.DIABETES = 1; 

  

  --Validate Diabetes and Hypertension View 

  SELECT * FROM HYDRATED_DIABETES_PATIENTS; 

  

  

-- View for Patients by Age Group and Star Sign 

CREATE OR REPLACE VIEW PATIENTS_BY_AGE_AND_STAR_SIGN AS 

SELECT DISTINCT 

    p.AGE_GROUP, 

    p.STAR_SIGN, 

     

    COUNT(p.PATIENT_ID) AS PATIENT_COUNT 

FROM MELLITUS_DEV_CURATED.PUBLIC.PATIENT_DIM p 

GROUP BY p.AGE_GROUP, p.STAR_SIGN 

ORDER BY PATIENT_COUNT DESC; 

  

  

--View for BMI and weight distribution 

  

CREATE OR REPLACE VIEW BMI_AND_WEIGHT_DISTRIBUTION AS 

SELECT DISTINCT 

    p.PATIENT_ID, 

    m.BMI, 

    m.WEIGHT, 

    CASE  

        WHEN m.BMI < 18.5 THEN 'Underweight' 

        WHEN m.BMI BETWEEN 18.5 AND 24.9 THEN 'Normal weight' 

        WHEN m.BMI BETWEEN 25 AND 29.9 THEN 'Overweight' 

        ELSE 'Obesity' 

    END AS BMI_CATEGORY 

FROM MELLITUS_DEV_CURATED.PUBLIC.PATIENT_DIM p 

JOIN MELLITUS_DEV_CURATED.PUBLIC.HEALTH_METRICS_DIM m  

    ON p.PATIENT_ID = m.PATIENT_ID; 

  

  

  

--Family History and Diabetes Risk 

  

CREATE OR REPLACE VIEW FAMILY_HISTORY_AND_DIABETES_RISK AS 

SELECT  

    p.PATIENT_ID, 

    m.FAMILY_DIABETES_HISTORY, 

    d.DIABETES, 

    COUNT(p.PATIENT_ID) AS PATIENT_COUNT 

FROM MELLITUS_DEV_CURATED.PUBLIC.PATIENT_DIM p 

JOIN MELLITUS_DEV_CURATED.PUBLIC.DIABETES_FACT d 

    ON p.PATIENT_ID = d.PATIENT_ID 

     

JOIN MELLITUS_DEV_CURATED.PUBLIC.HEALTH_METRICS_DIM m  

    ON p.PATIENT_ID = m.PATIENT_ID 

     

WHERE m.FAMILY_DIABETES_HISTORY = 1 

GROUP BY p.PATIENT_ID, m.FAMILY_DIABETES_HISTORY, d.DIABETES 

ORDER BY PATIENT_COUNT DESC; 

  

  

--View for Predicted Diabetes by Age Group and Gender 

  

CREATE OR REPLACE VIEW PREDICTED_DIABETES_BY_AGE_GENDER AS 

SELECT DISTINCT 

    p.AGE_GROUP, 

    p.GENDER, 

    d.PREDICTED_DIABETES_FLAG, 

    COUNT(p.PATIENT_ID) AS PATIENT_COUNT 

FROM MELLITUS_DEV_CURATED.PUBLIC.PATIENT_DIM p 

  

JOIN MELLITUS_DEV_CURATED.PUBLIC.DIABETES_FACT d 

    ON p.PATIENT_ID = d.PATIENT_ID 

     

GROUP BY p.AGE_GROUP, p.GENDER, d.PREDICTED_DIABETES_FLAG 

ORDER BY p.AGE_GROUP, d.PREDICTED_DIABETES_FLAG DESC; 

  

  

  

--View for Diabetes and Sleep Duration 

  

CREATE OR REPLACE VIEW DIABETES_AND_SLEEP_DURATION AS 

SELECT  

    p.PATIENT_ID, 

    l.SLEEP_DURATION, 

    CASE  

        WHEN l.SLEEP_DURATION < 6 THEN 'Low Sleep' 

        WHEN l.SLEEP_DURATION BETWEEN 6 AND 8 THEN 'Moderate Sleep' 

        ELSE 'High Sleep' 

    END AS SLEEP_CATEGORY, 

    d.DIABETES, 

    COUNT(p.PATIENT_ID) AS PATIENT_COUNT 

FROM MELLITUS_DEV_CURATED.PUBLIC.PATIENT_DIM p 

JOIN MELLITUS_DEV_CURATED.PUBLIC.LIFESTYLE_DIM l 

    ON p.PATIENT_ID = l.PATIENT_ID 

JOIN MELLITUS_DEV_CURATED.PUBLIC.DIABETES_FACT d 

    ON p.PATIENT_ID = d.PATIENT_ID 

GROUP BY p.PATIENT_ID, l.SLEEP_DURATION, d.DIABETES 

ORDER BY SLEEP_CATEGORY, PATIENT_COUNT DESC; 

  

  

  

---View to analyze the relationship between BMI and Diabetes 

  

CREATE OR REPLACE VIEW BMI_AND_DIABETES_CORRELATION AS 

SELECT  

    m.BMI, 

    CASE  

        WHEN m.BMI < 18.5 THEN 'Underweight' 

        WHEN m.BMI BETWEEN 18.5 AND 24.9 THEN 'Normal weight' 

        WHEN m.BMI BETWEEN 25 AND 29.9 THEN 'Overweight' 

        ELSE 'Obesity' 

    END AS BMI_CATEGORY, 

    d.DIABETES, 

    COUNT(d.PATIENT_ID) AS PATIENT_COUNT 

FROM MELLITUS_DEV_CURATED.PUBLIC.DIABETES_FACT d 

JOIN MELLITUS_DEV_CURATED.PUBLIC.HEALTH_METRICS_DIM m 

    ON d.METRIC_ID = m.METRIC_ID 

GROUP BY m.BMI, d.DIABETES 

ORDER BY 2, PATIENT_COUNT DESC; 

  

  

  

-- Drop Risk Column in diabetes_fact table 

  

ALTER TABLE MELLITUS_DEV_CURATED.PUBLIC.DIABETES_FACT 

DROP COLUMN RISK_SCORE; 

  

  

---View to analyze the relationship between Hypertension and Diabetes 

CREATE OR REPLACE VIEW DIABETES_AND_HYPERTENSION_CORRELATION AS 

SELECT  

    p.GENDER, 

    p.AGE_GROUP, 

    m.HYPERTENSION, 

    d.DIABETES, 

    COUNT(p.PATIENT_ID) AS PATIENT_COUNT 

FROM MELLITUS_DEV_CURATED.PUBLIC.PATIENT_DIM p 

JOIN MELLITUS_DEV_CURATED.PUBLIC.HEALTH_METRICS_DIM m 

    ON p.PATIENT_ID = m.PATIENT_ID 

JOIN MELLITUS_DEV_CURATED.PUBLIC.DIABETES_FACT d 

    ON d.PATIENT_ID = m.PATIENT_ID 

WHERE d.DIABETES = 1 

GROUP BY p.GENDER, p.AGE_GROUP, m.HYPERTENSION, d.DIABETES 

ORDER BY PATIENT_COUNT DESC; 

  

  

-- View for DIABETES RISK FACTORS AND lIFESTYLE 

  

CREATE OR REPLACE VIEW DIABETES_RISK_FACTORS_AND_LIFESTYLE AS 

SELECT DISTINCT 

    l.PATIENT_ID, 

    l.DIET_TYPE, 

    l.PHYSICAL_ACTIVITY_LEVEL, 

    l.SLEEP_DURATION, 

    l.STRESS_LEVEL, 

    d.DIABETES, 

    CASE 

        WHEN l.STRESS_LEVEL = 'High' THEN 1 

        ELSE 0 

    END AS HIGH_STRESS_FLAG, 

    CASE 

        WHEN l.SLEEP_DURATION < 6 THEN 'Low Sleep' 

        WHEN l.SLEEP_DURATION BETWEEN 6 AND 8 THEN 'Moderate Sleep' 

        ELSE 'High Sleep' 

    END AS SLEEP_CATEGORY 

FROM MELLITUS_DEV_CURATED.PUBLIC.LIFESTYLE_DIM l 

JOIN MELLITUS_DEV_CURATED.PUBLIC.DIABETES_FACT d 

    ON l.PATIENT_ID = d.PATIENT_ID 

WHERE d.DIABETES = 1; 

  

-- Diabetes Prevalence by Demographic Factors 

  

CREATE OR REPLACE VIEW DIABETES_BY_DEMOGRAPHICS AS 

SELECT DISTINCT 

    p.GENDER, 

    p.AGE_GROUP, 

    m.FAMILY_DIABETES_HISTORY, 

    COUNT(p.PATIENT_ID) AS DIABETES_COUNT, 

    AVG(d.DIABETES) AS DIABETES_PREVALENCE 

FROM MELLITUS_DEV_CURATED.PUBLIC.PATIENT_DIM p 

JOIN MELLITUS_DEV_CURATED.PUBLIC.DIABETES_FACT d 

ON p.PATIENT_ID = d.PATIENT_ID 

  

JOIN MELLITUS_DEV_CURATED.PUBLIC.HEALTH_METRICS_DIM m 

ON p.PATIENT_ID = m.PATIENT_ID 

     

GROUP BY p.GENDER, p.AGE_GROUP, m.FAMILY_DIABETES_HISTORY 

ORDER BY DIABETES_PREVALENCE DESC; 

  

--View for Lifestyle and Predicted Diabetes 

  

CREATE OR REPLACE VIEW LIFESTYLE_AND_PREDICTED_DIABETES AS 

SELECT  

    l.DIET_TYPE, 

    l.PHYSICAL_ACTIVITY_LEVEL, 

    d.PREDICTED_DIABETES_FLAG, 

    COUNT(d.PATIENT_ID) AS PATIENT_COUNT 

FROM MELLITUS_DEV_CURATED.PUBLIC.DIABETES_FACT d 

JOIN MELLITUS_DEV_CURATED.PUBLIC.LIFESTYLE_DIM l 

    ON d.PATIENT_ID = l.PATIENT_ID 

GROUP BY l.DIET_TYPE, l.PHYSICAL_ACTIVITY_LEVEL, d.PREDICTED_DIABETES_FLAG; 

  

  

-- View for Alcohol Consumption and Family History of Diabetes 

  

CREATE OR REPLACE VIEW ALCOHOL_AND_FAMILY_HISTORY AS 

SELECT  

    p.PATIENT_ID, 

    l.ALCOHOL_CONSUMPTION, 

    m.FAMILY_DIABETES_HISTORY, 

    d.DIABETES, 

    p.AGE, 

    p.AGE_GROUP 

FROM MELLITUS_DEV_CURATED.PUBLIC.PATIENT_DIM p 

JOIN MELLITUS_DEV_CURATED.PUBLIC.LIFESTYLE_DIM l 

    ON p.PATIENT_ID = l.PATIENT_ID 

JOIN MELLITUS_DEV_CURATED.PUBLIC.HEALTH_METRICS_DIM m 

    ON p.PATIENT_ID = m.PATIENT_ID 

JOIN MELLITUS_DEV_CURATED.PUBLIC.DIABETES_FACT d 

    ON p.PATIENT_ID = d.PATIENT_ID 

WHERE l.ALCOHOL_CONSUMPTION = 'High'  

  AND m.FAMILY_DIABETES_HISTORY = 1; 

  

--View for Health conditions and Stress 

  

CREATE OR REPLACE VIEW HEALTH_CONDITIONS_AND_STRESS AS 

SELECT  

    p.PATIENT_ID, 

    m.HYPERTENSION, 

    d.DIABETES, 

    l.STRESS_LEVEL, 

    COUNT(p.PATIENT_ID) AS PATIENT_COUNT 

FROM MELLITUS_DEV_CURATED.PUBLIC.PATIENT_DIM p 

JOIN MELLITUS_DEV_CURATED.PUBLIC.LIFESTYLE_DIM l 

    ON p.PATIENT_ID = l.PATIENT_ID 

JOIN MELLITUS_DEV_CURATED.PUBLIC.HEALTH_METRICS_DIM m 

    ON p.PATIENT_ID = m.PATIENT_ID 

JOIN MELLITUS_DEV_CURATED.PUBLIC.DIABETES_FACT d 

    ON p.PATIENT_ID = d.PATIENT_ID 

GROUP BY m.HYPERTENSION, d.DIABETES, l.STRESS_LEVEL, p.PATIENT_ID 

ORDER BY m.HYPERTENSION DESC, d.DIABETES DESC, l.STRESS_LEVEL DESC; 

  

  

--Create View of Predicted Diabetes by Agegroup 

  

CREATE OR REPLACE VIEW PREDICTED_DIABETES_BY_AGE AS 

SELECT DISTINCT 

    p.AGE_GROUP, 

    d.PREDICTED_DIABETES_FLAG, 

    COUNT(p.PATIENT_ID) AS PATIENT_COUNT 

FROM MELLITUS_DEV_CURATED.PUBLIC.PATIENT_DIM p 

  

JOIN MELLITUS_DEV_CURATED.PUBLIC.DIABETES_FACT d 

ON p.PATIENT_ID = d.PATIENT_ID 

     

GROUP BY p.AGE_GROUP, d.PREDICTED_DIABETES_FLAG 

ORDER BY p.AGE_GROUP, d.PREDICTED_DIABETES_FLAG DESC; 

  

-- Create View of Diabetes Feature Importance 

  

CREATE VIEW MELLITUS_DEV_SERVING.PUBLIC.DIABETES_FEATURE_IMPORTANCE_VIEW AS 

SELECT * 

FROM MELLITUS_DEV_CURATED.MODELLING.DIABETES_FEATURE_IMPORTANCE; 

  

  

--Create View for Diabetes Model Evaluation 

  

CREATE VIEW MELLITUS_DEV_SERVING.PUBLIC.DIABETES_MODEL_EVALUATION_VIEW AS 

SELECT * 

FROM MELLITUS_DEV_CURATED.MODELLING.DIABETES_MODEL_EVALUATION; 