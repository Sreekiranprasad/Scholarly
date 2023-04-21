A scholar recommendation tool to identify potential scholars in the industry

<br />


## PROJECT DESCRIPTION

 - The tool extracts and creates user/scholar profile using the TAMU scholars library using APIs
 - Matches and recommends scholars based on user query or scholar matching algorithm

<br />


## Workflow - Recommending Scholars Queries



<br />

## Usage Instructions
<br />




Step 1 : Create list of Scholars

```
python user_profile_creation.py --univ_name='TAMU'
```

Step 2 : Create publication database

```
python extract_publications.py --n_cores=20
```

Step 3 : Create Analytical database

```
python create_analytical_data.py --n_cores=20
```
