# Organizational Intelligence Platform for Institutional Research(In Progress)

<br />


## PROJECT DESCRIPTION

 - The tool extracts and creates user/scholar profile using the TAMU scholars library using APIs
 - Matches and recommends user profile to research proposals
 - Identify similar research profiles for each scholar
 - Advance Oppurtunities for Intelligent Research
 - Recommend latest relevant articles/publications for literature searcha and advancement

<br />


## Workflow - Recommending Scholars for Proposals

![](Misc/RDash%20-%20Workflow.png)

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

Step 4 : Compile list of Grants

```
python extract_proposals.py 
```

Step 4 : Extract grant details

```
python main_extractor.py --n_cores=20 --a 'National Science Foundation' 'National Institutes of Health'
```

Step 5 : Recommend scholars for a Proposal / grant

```
python recommend_scholars.py --top_k=20 --proposal_id='PD-18-1263' --n_cores=20 --agency='NSF'
```

Step 6: Extract proposals to a json for searching

```
python extract_proposals_titles_db.py
```

<br />

## To Host Server (via Docker)

NB : If running from datahub append 'sudo' before each command below
```
docker build --tag rdash_backend .
docker run --name rdash_backend -p 5001:9000  rdash_backend
```

To run in detached mode use:
```
docker run --name rdash_backend -p 5001:9000 -d rdash_backend
```
Here I am using 5001 port so that the api is accessible at http://datahub.geos.tamu.edu:5001/. Feel free to change this port and if it is changed here update the port in WebApp as well.

It takes around 10 mins to load all the datasets and after which the application starts running.
you can inspect the logs meanwhile using:

```
docker logs <container_id>
```

Also you can reach to the endpoint http://datahub.geos.tamu.edu:5001/test/ to get a "hi" response.

