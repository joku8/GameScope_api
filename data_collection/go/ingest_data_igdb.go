package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
	"github.com/joho/godotenv"
)

const (
	baseURL     = "https://api.igdb.com/v4"
	tokenURL    = "https://id.twitch.tv/oauth2/token"
	batchSize   = 500
	rateLimitMs = 300 // ~3 req/sec cuz i think it only allows 4 req/sec
)

type Game struct {		// need to change these maybe?
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

func getAccessToken(clientID, clientSecret string) (string, error) {
	req, err := http.NewRequest("POST", tokenURL, nil)
	if err != nil {
		return "", err
	}

	q := req.URL.Query()
	q.Add("client_id", clientID)
	q.Add("client_secret", clientSecret)
	q.Add("grant_type", "client_credentials")
	req.URL.RawQuery = q.Encode()

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var data struct {
		AccessToken string `json:"access_token"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return "", err
	}

	return data.AccessToken, nil
}

func fetchGames(client *http.Client, token string, lastID int) ([]Game, error) {
	query := fmt.Sprintf(`
	fields id, name, first_release_date, genres, platforms, themes,
	       rating, rating_count, total_rating, total_rating_count,
	       summary, storyline;
	where id > %d & first_release_date != null;
	sort id asc;
	limit %d;
	`, lastID, batchSize)

	req, err := http.NewRequest("POST", baseURL+"/games", bytes.NewBufferString(query))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Client-ID", os.Getenv("IGDB_CLIENT_ID"))
	req.Header.Set("Authorization", "Bearer " + token)

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	var games []Game
	if err := json.Unmarshal(body, &games); err != nil {
		return nil, err
	}

	return games, nil
}

func formatDate(ts int64) string {
	if ts == 0 {
		return ""
	}
	return time.Unix(ts, 0).UTC().Format("2006-01-02")
}

func joinInts(ints []int) string {
	if len(ints) == 0 {
		return ""
	}
	s := make([]string, len(ints))
	for i, v := range ints {
		s[i] = strconv.Itoa(v)
	}
	return strings.Join(s, ",")
}

func main() {
	if err := godotenv.Load("../../.env"); err != nil {
        fmt.Println("Warning: .env file not found, relying on environment variables")
    }
	clientID := os.Getenv("IGDB_CLIENT_ID")
	clientSecret := os.Getenv("IGDB_CLIENT_SECRET")

	if clientID == "" || clientSecret == "" {
		panic("Missing IGDB_CLIENT_ID or IGDB_CLIENT_SECRET")
	}

	fmt.Println("Getting access token...")
	token, err := getAccessToken(clientID, clientSecret)
	if err != nil {
		panic(err)
	}


	fmt.Println("Token acquired")

	file, err := os.Create("../out/igdb_games_all.csv")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	header := []string{
		"id", "name", "first_release_date",
		"genres", "platforms", "themes",
		"rating", "rating_count",
		"total_rating", "total_rating_count",
		"summary", "storyline",
	}
	writer.Write(header)

	httpClient := &http.Client{Timeout: 30 * time.Second}

	lastID := 0
	total := 0

	for {
		games, err := fetchGames(httpClient, token, lastID)
		if err != nil {
			panic(err)
		}
		if len(games) == 0 {
			break
		}
		for _, g := range games {
			lastID = g.ID
			total++
			row := []string{
				strconv.Itoa(g.ID),
				g.Name,
				formatDate(g.FirstReleaseDate),
				joinInts(g.Genres),
				joinInts(g.Platforms),
				joinInts(g.Themes),
				fmt.Sprintf("%.2f", g.Rating),
				strconv.Itoa(g.RatingCount),
				fmt.Sprintf("%.2f", g.TotalRating),
				strconv.Itoa(g.TotalRatingCount),
				g.Summary,
				g.Storyline,
			}
			writer.Write(row)
		}

		writer.Flush()
		fmt.Printf("Fetched %d games (last id: %d)\n", total, lastID)

		time.Sleep(rateLimitMs * time.Millisecond)
	}

	fmt.Println("Done.")
}
