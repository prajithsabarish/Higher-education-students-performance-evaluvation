# Dataset Documentation: Higher Education Students' Performance Evaluation

## 1. Overview and Provenance
- **Dataset Name:** Higher Education Students Performance Evaluation
- **Source:** UCI Machine Learning Repository
- **Authors:** Nevruz Yilmaz and Boran Sekeroglu (2019)
- **Institution:** Department of Computer Engineering, Near East University, Nicosia, Northern Cyprus
- **Academic Citation:**
  > Yilmaz, N., & Sekeroglu, B. (2019). Student Performance Evaluation using Machine Learning Techniques. *International Journal of Advanced Trends in Computer Science and Engineering*, 8(1.2), 200–207.
- **Repository Link:** [UCI Machine Learning Repository - Higher Education Students Performance Evaluation](https://archive.ics.uci.edu/dataset/856/higher+education+students+performance+evaluation)

---

## 2. Cohort Details & Tabular Geometry
- **Total Records:** 145 students
- **Total Columns:** 33 features
  - 1 Identifier (`STUDENT ID`: STUDENT1 to STUDENT145)
  - 30 Survey Questions (Headers originally '1' to '30')
  - 1 Institutional Context Variable (`COURSE ID`: 1 to 9)
  - 1 Target Outcome (`GRADE`: 0 to 7)
- **Missing Values:** 0 (100% complete)
- **Duplicate Rows:** 0 (full matrix and excluding Student ID)

---

## 3. Data Dictionary & Attribute Encoding Scheme

| Column # | Renamed Feature | Original Question / Concept | Encoded Integer Range & Decoded Values |
| :---: | :--- | :--- | :--- |
| **0** | `STUDENT ID` | Student Identification Code | Arbitrary string: `STUDENT1` to `STUDENT145` |
| **1** | `Age` | Student age category | **1:** 18-21, **2:** 22-25, **3:** 26+ |
| **2** | `Sex` | Biological sex | **1:** Female, **2:** Male |
| **3** | `HS_Type` | High school graduation type | **1:** Private, **2:** State, **3:** Other |
| **4** | `Scholarship` | University tuition scholarship tier | **1:** None (0%), **2:** 25%, **3:** 50%, **4:** 75%, **5:** Full (100%) |
| **5** | `Additional_Work` | Employment during academic term | **1:** Yes, **2:** No |
| **6** | `Activity` | Regular artistic or sports activity | **1:** Yes, **2:** No |
| **7** | `Partner` | Has romantic partner / relationship | **1:** Yes, **2:** No |
| **8** | `Salary` | Monthly disposable allowance / income | **1:** USD 135-200, **2:** USD 201-270, **3:** USD 271-340, **4:** USD 341-410, **5:** >USD 410 |
| **9** | `Transport` | Primary campus commute mode | **1:** Bus, **2:** Car/Taxi, **3:** Bicycle, **4:** Other |
| **10** | `Accommodation` | Living arrangement during term | **1:** Rental apartment, **2:** Dormitory, **3:** With Family, **4:** Other |
| **11** | `Mother_Edu` | Mother's highest completed education | **1:** Primary, **2:** Secondary, **3:** High School, **4:** University, **5:** MSc, **6:** PhD |
| **12** | `Father_Edu` | Father's highest completed education | **1:** Primary, **2:** Secondary, **3:** High School, **4:** University, **5:** MSc, **6:** PhD |
| **13** | `Siblings` | Number of living siblings | **1:** 1, **2:** 2, **3:** 3, **4:** 4, **5:** 5+ |
| **14** | `Parental_Status` | Parents' current marital status | **1:** Married, **2:** Divorced, **3:** Died (one or both) |
| **15** | `Mother_Occ` | Mother's employment category | **1:** Retired, **2:** Housewife, **3:** Government Officer, **4:** Private Employee, **5:** Self-Employed, *(6: Other - 0 occurrences)* |
| **16** | `Father_Occ` | Father's employment category | **1:** Retired, **2:** Government Officer, **3:** Private Employee, **4:** Self-Employed, **5:** Other |
| **17** | `Study_Hours` | Weekly self-study duration | **1:** None, **2:** <5 hours, **3:** 6-10 hours, **4:** 11-20 hours, **5:** >20 hours |
| **18** | `Read_NonSci` | Non-scientific reading frequency | **1:** None, **2:** Sometimes, **3:** Often |
| **19** | `Read_Sci` | Scientific books/articles reading | **1:** None, **2:** Sometimes, **3:** Often |
| **20** | `Seminars` | Academic seminars / conferences attendance | **1:** Yes, **2:** No |
| **21** | `Project_Impact` | Impact of practical projects on learning | **1:** Positive, **2:** Negative, **3:** Neutral |
| **22** | `Attendance` | Class lecture attendance compliance | **1:** Always, **2:** Sometimes, *(3: Never - 0 occurrences)* |
| **23** | `Midterm_Prep_With` | Peer study partner for midterms | **1:** Alone, **2:** With Friends, **3:** Not Applicable |
| **24** | `Midterm_Prep_Timing` | Exam study timing habit | **1:** Closest to Exam (procrastination), **2:** Regularly, **3:** Never |
| **25** | `Note_Taking` | In-class note-taking frequency | **1:** Never, **2:** Sometimes, **3:** Always |
| **26** | `Listening` | Active listening during lectures | **1:** Never, **2:** Sometimes, **3:** Always |
| **27** | `Discussion` | Classroom discussion participation | **1:** Never, **2:** Sometimes, **3:** Always |
| **28** | `Flip_Classroom` | Utility perception of flipped classroom | **1:** Not Useful, **2:** Useful, **3:** Not Applicable |
| **29** | `GPA_Last` | Prior cumulative GPA standing | **1:** <2.00, **2:** 2.00-2.49, **3:** 2.50-2.99, **4:** 3.00-3.49, **5:** >3.49 |
| **30** | `GPA_Expected` | Expected end-of-term GPA standing | **1:** <2.00, **2:** 2.00-2.49, **3:** 2.50-2.99, **4:** 3.00-3.49, *(5: >3.49 - 0 occurrences)* |
| **31** | `Course_ID` | Course Offering Code | **1 to 9** (Course 1: 66, Course 2: 2, Course 3: 8, Course 4: 4, Course 5: 7, Course 6: 8, Course 7: 15, Course 8: 14, Course 9: 21) |
| **32** | `Grade` | Final End-of-Term Grade (Target) | **0:** Fail (8), **1:** DD (35), **2:** DC (24), **3:** CC (21), **4:** CB (10), **5:** BB (17), **6:** BA (13), **7:** AA (17) |

---

## 4. Derived & Engineered Variables
In addition to the raw items, the research pipeline constructs the following composite indices:
1. **`Study_Engagement`** = $\text{Study\_Hours} + \text{Read\_Sci} + \text{Read\_NonSci} + \text{Seminars\_Scored}$ (where Seminars: Yes=2, No=1)
2. **`Class_Behaviour`** = $(4 - \text{Attendance}) + \text{Note\_Taking} + \text{Listening} + \text{Discussion}$ (where Attendance is reverse-coded: Always=3, Sometimes=2, Never=1)
3. **`Family_Education`** = $\text{Mother\_Edu} + \text{Father\_Edu}$
4. **`Aspiration_Gap`** = $\text{GPA\_Expected} - \text{GPA\_Last}$
5. **`Grade_Group`** = Categorical risk tier:
   - **Low Attainment:** Grades 0 to 2 (Fail, DD, DC; $n=67, 46.2\%$)
   - **Medium Attainment:** Grades 3 to 4 (CC, CB; $n=31, 21.4\%$)
   - **High Attainment:** Grades 5 to 7 (BB, BA, AA; $n=47, 32.4\%$)
6. **`Pass`** = $\mathbb{I}(\text{Grade} \ge 1)$ ($n=137$ Pass, $n=8$ Fail)
