# Group 4: Movie Recommendation System

A movie recommender built on the MovieLens "small" dataset. Given a user's ratings of other movies, it returns **5 movies the user has not seen and is likely to rate highly**.

## 1. Business Problem

Streaming platforms host thousands of titles, and users struggle to find something worth watching. Poor discovery lowers engagement and increases churn. This project helps a streaming service show each user a short, personalised list of 5 unseen movies.

**How a user provides ratings:** as a Python dictionary of movie title and rating, for example:

```python
{"Toy Story (1995)": 5, "Heat (1995)": 3, "Jumanji (1995)": 4}
```

## 2. Data

- **Source:** MovieLens `ml-latest-small` from the GroupLens research lab, University of Minnesota
- **Link:** https://grouplens.org/datasets/movielens/latest/
- **Size:** about 100,000 ratings from 610 users on about 9,700 movies
- **Files used:** `ratings.csv` (collaborative filtering) and `movies.csv` (titles and genres for cold start)
- **Files not used:** `tags.csv` and `links.csv` (possible extension)

The dataset is **not stored in this repo**. The notebook downloads it automatically on first run.

## 3. Approach (CRISP-DM)

1. **Business Understanding:** define the problem, success criteria and how users supply ratings
2. **Data Understanding:** sparsity, rating distribution, ratings per user and per movie, genres
3. **Data Preparation:** train/test split, user-item matrix built from training data only
4. **Modeling:**
   - Baseline: movie mean rating
   - Item-based collaborative filtering (cosine similarity on mean-centered ratings)
   - SVD matrix factorisation (comparison)
   - Cold-start hybrid: popularity for new users, genre-based similarity for new movies
5. **Evaluation:** RMSE and MAE on the test set against the baseline, optionally Precision@5
6. **Deployment:** `recommend_top5(user_ratings)` function with a demo user

## 4. Results

_To be filled in after the models are run._

| Model | RMSE | MAE |
|---|---|---|
| Baseline (movie mean) | | |
| Item-based CF | | |
| SVD | | |

## 5. Repository Structure

```
GROUP-4-MOVIE-RECOMMENDATION-SYSTEM/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    ├── 01_business_and_data_understanding.ipynb
    ├── 02_data_preparation.ipynb
    ├── 03_modeling.ipynb
    └── 04_evaluation_and_deployment.ipynb
```

## 6. How to Run

```bash
git clone https://github.com/Tao812-creator/GROUP-4-MOVIE-RECOMMENDATION-SYSTEM.git
cd GROUP-4-MOVIE-RECOMMENDATION-SYSTEM
pip install -r requirements.txt
jupyter notebook
```

Open the notebooks in order. The first notebook downloads the dataset.

## 7. Team and Roles

| Member | GitHub | Responsibility |
|---|---|---|
| _Name_ | _username_ | Business and Data Understanding (intro, EDA) |
| _Name_ | _username_ | Data Preparation |
| _Name_ | _username_ | Baseline and item-based CF |
| _Name_ | _username_ | SVD and cold-start hybrid |
| Everyone | | Evaluation, `recommend_top5`, README |

## 8. Collaboration Workflow

- Never commit directly to `main`. Work on your own branch (e.g. `yourname-eda`).
- Run `git pull` before you start working.
- Open a Pull Request and have a teammate review it before merging.
- Clear notebook outputs before committing to keep diffs small.
- Each person works in their own notebook to avoid merge conflicts.

## 9. Limitations and Next Steps

- Genres alone are coarse for cold-start recommendations; adding `tags.csv` could improve them.
- Item-based similarity is unreliable for movies with very few ratings.
- Ideas to try next: tuning SVD, hybrid weighting, ranking metrics beyond Precision@5.
