# Movie Recommendation System (MovieLens)

## Project overview

This group project develops an offline movie recommendation prototype using the MovieLens small dataset. Given a user's ratings for movies already watched, the system returns five movies that the user has not rated.

The solution combines:

- **Item-based collaborative filtering**, which identifies movies with similar user-rating patterns.
- **Content-based filtering**, which compares movie genres using TF-IDF features and cosine similarity.
- **A hybrid recommendation approach**, which combines collaborative and genre-based signals.

The notebook follows the **CRISP-DM** workflow: business understanding, data understanding, data preparation, modelling, evaluation and deployment. The final recommendation function is named `recommend()`.

This is an offline academic prototype. It is not connected to a live streaming platform, production database or real-time user activity.

## Dataset

The project uses the MovieLens small dataset, which contains:

- `movies.csv` — movie IDs, titles and genre labels.
- `ratings.csv` — user IDs, movie IDs, ratings and timestamps.

The files loaded in the notebook contain 9,742 movies and 100,836 ratings. The rating scale ranges from 0.5 to 5 stars.

Dataset source: [GroupLens – MovieLens datasets](https://grouplens.org/datasets/movielens/)

Download the **MovieLens Latest Small** dataset and place `movies.csv` and `ratings.csv` in the same directory as the notebook.

## Project structure

```text
movie-recommendation-system/
├── movie_recommendation_system_group_report.ipynb
├── README_movie_recommendation_system.md
├── movies.csv
└── ratings.csv
```

The CSV files are not embedded in the notebook. They must be available in the notebook's working directory before the data-loading cells are run.

## Requirements

The notebook uses Python and the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn

Install the dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
```

## How to run

1. Download and extract the MovieLens Latest Small dataset.
2. Place `movies.csv` and `ratings.csv` in the same folder as the notebook.
3. Install the required Python libraries.
4. Open the notebook in Jupyter Notebook or JupyterLab.
5. Run the cells from top to bottom so that the data, model components and recommendation function are created in the correct order.
6. Review the evaluation tables and example recommendations near the end of the notebook.

## Methodology

### 1. Data understanding

The notebook examines dataset dimensions, missing values, duplicate user–movie pairs, rating distributions, ratings per user, ratings per movie and genre frequencies. The rating matrix is approximately 98.30% sparse, meaning that most possible user–movie pairs have no recorded rating.

### 2. Data preparation

The ratings are divided into an 80% training set and a 20% test set using `random_state=42`. User and movie IDs are mapped to matrix indices. A baseline rating is calculated from the global mean, user bias and item bias. Residuals are calculated as actual ratings minus baseline predictions.

Movies with at least five ratings in the training set are included in the collaborative-filtering candidate set. Movie genres are converted into TF-IDF vectors for content-based comparisons.

### 3. Modelling

The notebook compares a global-mean baseline, a bias baseline, item-based collaborative filtering, genre-based content filtering and a hybrid model. Item-based collaborative filtering uses cosine similarity between residual rating patterns. Similarities are adjusted according to the number of users who rated both movies. Up to 30 neighbouring movies are used in residual estimates.

The hybrid prediction combines the baseline with collaborative and content-based residual estimates:

```text
Prediction = baseline + w × CF residual + (1 − w) × content residual
```

For movies outside the collaborative-filtering candidate set, the content-based residual is used because collaborative similarity is unavailable.

### 4. Evaluation

Rating prediction is evaluated with **RMSE** and **MAE**; lower values indicate better prediction accuracy. Top-five ranking is evaluated with **Precision@5**, where a test rating of 4 or higher is treated as relevant. The ranking evaluation includes 596 users who have at least one positively rated test movie within the candidate set.

The test set is also used when comparing hybrid weights and popularity boosts. As noted in the notebook, using the test set for parameter selection can make the reported results somewhat optimistic; a separate validation set would provide a more rigorous evaluation.

## Key results reported in the notebook

### Rating prediction

| Model | RMSE | MAE |
|---|---:|---:|
| Global mean | 1.0488 | 0.8316 |
| Bias baseline | 0.8932 | 0.6810 |
| Item-based collaborative filtering | 0.8573 | 0.6499 |
| Content-based genres | 0.8600 | 0.6572 |
| Hybrid (`w = 0.7`) | **0.8467** | **0.6441** |

The hybrid model with `w = 0.7` has the lowest RMSE and MAE among the rating-prediction models compared.

### Top-five recommendation ranking

| Recommendation approach | Precision@5 |
|---|---:|
| Random recommendations | 0.0057 |
| Most-rated movies | 0.1493 |
| Item-based collaborative filtering | 0.0782 |
| Hybrid (`w = 0.7`) | 0.0775 |
| Hybrid + popularity boost (0.3) | **0.1607** |

The hybrid model alone does not outperform the most-rated baseline on Precision@5. The hybrid model with a popularity boost of 0.3 achieves the highest Precision@5 among the approaches tested.

### Low-history movie evaluation

For 2,995 test ratings associated with movies outside the collaborative-filtering candidate set:

- Bias-baseline RMSE: 0.9909
- Genre-assisted hybrid RMSE: 0.9451

The lower RMSE indicates that genre information improves rating prediction for this low-history movie subset compared with the bias baseline alone.

## Recommendation function and examples

The `recommend()` function accepts a dictionary of movie titles and ratings, selects a recommendation mode, excludes movies already rated by the user and returns five candidate movies with their genres and predicted ratings.

The implementation uses a content-based-only fallback when the user provides fewer than five ratings, or fewer than three of the supplied rated movies belong to the collaborative-filtering candidate set. The notebook includes example profiles for science-fiction/action preferences, family/animation preferences and a new user with only two ratings.

## Limitations

- Genre labels provide limited information about movie content and user preferences.
- The top-five candidate list is restricted to movies with at least five training ratings.
- Hybrid and popularity parameters were selected using test-set results rather than a separate validation set.
- Offline test ratings are only a proxy for user satisfaction; a live experiment would be needed to assess actual engagement.
- The new-user fallback still requires some initial ratings.

## Possible future improvements

- Introduce a separate validation set for parameter tuning.
- Compare the current approach with matrix factorisation methods such as SVD.
- Add richer movie metadata, including tags, plot descriptions, cast and directors.
- Expand the candidate-generation strategy to include movies with very limited rating history.
- Conduct a user study or A/B test to evaluate real-world recommendation relevance.

## Group project note

This repository documents the shared project workflow, implementation and findings. The notebook uses neutral, report-style language to describe the work collectively rather than attributing individual tasks to one group member.
