# EduShield Uganda

**Keeping children in school by spotting dropout risk early and recommending what to do about it.**

EduShield Uganda is a web-based tool that helps teachers, school administrators, NGOs, and government officials identify which children (aged 3 to 10) are most at risk of dropping out of school — and tells you exactly what kind of help each child needs.

It uses real data from Uganda's national surveys and a machine learning model to answer two simple questions:

1. **Which children are most likely to drop out?**
2. **What can we do to keep them in school?**

---

## What the app does

When you open EduShield Uganda, you'll see four main sections:

### Overview — The big picture

A summary of the children in the dataset: how many there are, how many are at risk, and how risk breaks down by region, age, gender, and family wealth. Think of it as a "health check" for education across Uganda.

### Risk Prediction — Check on one child

Enter a child's details (age, gender, where they live, family size, poverty status) and the app will tell you:
- Whether the child is at **high**, **medium**, or **low** risk of dropping out
- A percentage showing how likely dropout is
- A list of recommended actions — for example, school feeding, cash support, transport help, or a girls' mentorship programme — ranked by urgency

### Regional Analytics — Compare across regions

See how dropout risk differs between Central, Eastern, Northern, and Western Uganda. Compare rural vs. urban areas. View UNESCO trends showing how Uganda compares to its East African neighbours over time.

### Early Warning — Children who need help now

A list of the children with the highest risk scores who need immediate attention, along with estimates of how many children need each type of intervention (feeding programmes, scholarships, transport, etc.).

---

## Where the data comes from

EduShield Uganda is built on official, publicly available data from four sources:

| Source | What it tells us |
|--------|-----------------|
| **Uganda National Panel Survey (UNPS) 2019/20** | Household-level information — income, location, family size, poverty status. This is the core dataset the model learns from. |
| **Uganda Demographic & Health Survey (UDHS) 2022** | Child health, nutrition, immunisation rates, and school attendance. Used to estimate realistic age and gender distributions. |
| **Uganda National Household Survey (UNHS) 2019/20** | Education enrolment, dropout rates, and the main reasons families give for why children leave school (cost, distance, child labour, early marriage, illness). |
| **UNESCO Institute for Statistics (UIS)** | International education indicators, including out-of-school rates and gender parity trends across East Africa. |

**How the child-level data is created:** The UNPS survey collects data at the household level, not for individual children. EduShield estimates the number of children aged 3-10 in each household (based on household size and Uganda's demographic profile), then assigns realistic ages and genders. This means the dataset is modelled — not a direct list of real children — but it faithfully reflects the patterns in the original survey data.

---

## How the risk score works

Each child gets a dropout risk score between 0% and 100%, based on six factors that research shows matter most:

| Factor | Weight | Why it matters |
|--------|--------|---------------|
| Poverty | 30% | 36% of dropouts cite cost as the main reason for leaving school |
| Wealth quintile | 20% | Families in the bottom 40% of wealth struggle to afford even "free" education |
| Rural location | 20% | 14% of dropouts cite distance to school; rural areas also face teacher shortages |
| Household size | 15% | Large families (7+ people) spread limited income across many children |
| Age | 10% | Younger children (3-5) face higher risk of never starting school at all |
| Gender | 5% | Girls face extra barriers like early marriage and lack of sanitary supplies |

A machine learning model (Histogram Gradient Boosting) is trained on this data each time the app starts, and typically achieves around 85-90% accuracy.

**Risk levels in plain English:**
- **High risk (70%+):** This child is very likely to drop out. Immediate action is needed.
- **Medium risk (40-70%):** This child could go either way. Preventive support now can make a big difference.
- **Low risk (below 40%):** This child is more likely to stay in school, but should still be monitored.

---

## What interventions are recommended

The app doesn't just flag risk — it tells you what to do. Recommendations are tailored to each child's specific situation:

| Situation | Recommended action | Why |
|-----------|--------------------|-----|
| Family is poor | School feeding programme; cash support (UGX 50,000/term) | Hungry children can't concentrate; families need help with hidden school costs |
| Child lives in a rural area | Transport help (bicycle or fare); mobile learning kits | Long walks to school and teacher shortages are major barriers |
| Large family (7+ people) | Full scholarship; family support services | Resources are stretched too thin across many children |
| Very low income (bottom 40%) | Free school supplies (books, uniforms, pens); parent skills training | Families can't afford basic materials; long-term income support helps |
| Girl | Girls' mentorship programme; hygiene and sanitary supplies | Girls miss school due to periods and face pressure toward early marriage |
| Older child (8-10) | Child labour prevention programme | Older children are pulled into farm or domestic work |

---

## How to set up and run the app

### What you'll need

- **Python 3.9 or newer** installed on your computer
- **The data files** in the `data/raw/` folder (see project structure below)
- About 10 minutes for first-time setup

### Step-by-step instructions

**1. Get the code**

```bash
git clone https://github.com/rryesuafuga/edushield-uganda.git
cd edushield-uganda
```

**2. Create a virtual environment** (recommended, keeps things tidy)

```bash
python -m venv venv
source venv/bin/activate        # On Mac/Linux
venv\Scripts\activate           # On Windows
```

**3. Install the required packages**

```bash
pip install -r requirements.txt
```

**4. Make sure the data is in place**

The app expects data files at these locations:

```
data/
  raw/
    UGA_2019_UNPS_v03_M_CSV/
      pov2019_20.csv              <-- Required (household poverty data)
    indicator-data-export_ROFST.1T3.GPIA.CP/
      data.json                   <-- Required (UNESCO education trends)
```

If you don't have these files, you can download the UNPS 2019/20 data from the [World Bank Microdata Library](https://microdata.worldbank.org/) and the UNESCO data from [UNESCO UIS](http://data.uis.unesco.org/).

**5. Run the app**

```bash
streamlit run app/dashboard.py
```

The app will open in your web browser (usually at `http://localhost:8501`). The first load takes a minute or two while the model trains — after that, it's cached and loads quickly.

---

## Deploying online

You can deploy EduShield Uganda for free on **Streamlit Community Cloud**:

1. Push this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account and select this repository
4. Set the main file path to `app/dashboard.py`
5. Click **Deploy**

Other options include [Render](https://render.com), [Railway](https://railway.app), or any platform that supports Python web apps.

---

## Project structure

```
edushield-uganda/
├── app/
│   ├── dashboard.py        # The main web interface (Streamlit)
│   └── recommender.py      # Intervention logic and national statistics
├── data/
│   └── raw/                # Source data files (not tracked in Git)
│       ├── UGA_2019_UNPS_v03_M_CSV/   # UNPS household survey
│       └── indicator-data-export.../   # UNESCO education indicators
├── models/
│   └── dropout_predictor.pkl   # Trained model (generated at runtime)
├── requirements.txt        # Python packages needed
├── .gitignore
└── README.md
```

---

## Tech stack

| Tool | What it's used for |
|------|--------------------|
| **Streamlit** | The web framework that powers the dashboard |
| **scikit-learn** | The machine learning model that predicts dropout risk |
| **pandas & NumPy** | Data processing and analysis |
| **Plotly** | Interactive, professional charts and infographics |
| **matplotlib & seaborn** | Additional data visualisation |

---

## Key numbers at a glance

These come from Uganda's official surveys and reports:

- **716,000** children of primary age are out of school in Uganda (UNESCO, 2022)
- **20.3%** of Ugandans live below the poverty line (UNHS 2019/20)
- **36%** of school dropouts say cost was the main reason they left
- **14%** cite distance to school
- **12%** cite child labour
- **43:1** is the national pupil-to-teacher ratio in primary schools
- **13.6%** of children aged 3-5 attend early childhood education

---

## Disclaimer

This tool is for **educational and policy support purposes only**. It does not store or expose personally identifiable information. The child-level records are modelled estimates derived from household survey data, not records of real individual children.

---

## Author

**Raymond Wayesu** — Biostatistician & Data Scientist

Data sources: UNPS 2019/20 | UDHS 2022 | UNHS 2019/20 | UNESCO UIS
