# Phase 0: Data Collection

We will use the [IGDB (Internet Game Database) by Twitch](https://www.igdb.com/) as our initial dataset.

The go script located at `./go/ingest_data_igdb.go` creates a csv with the following structure:

```go
type Game struct {
	ID                 int     `json:"id"`
	Name               string  `json:"name"`
	FirstReleaseDate   int64   `json:"first_release_date"`
	Genres             []int   `json:"genres"`
	Platforms          []int   `json:"platforms"`
	Themes             []int   `json:"themes"`
	Rating             float64 `json:"rating"`
	RatingCount        int     `json:"rating_count"`
	TotalRating        float64 `json:"total_rating"`
	TotalRatingCount   int     `json:"total_rating_count"`
	Summary            string  `json:"summary"`
	Storyline          string  `json:"storyline"`
}
```

In the future we may create some ingest to automatically pull in the latest and greatest data, and potentially from multiple sources. This will get us off the ground though.
