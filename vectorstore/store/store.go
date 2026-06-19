package store

import (
	"encoding/json"
	"math"
	"os"
	"sort"
	"sync"
)

// Entry holds a vector with its metadata.
type Entry struct {
	ID       string                 `json:"id"`
	Vector   []float32              `json:"vector"`
	Metadata map[string]interface{} `json:"metadata"`
}

// SearchResult is a single match from Search.
type SearchResult struct {
	Entry    Entry   `json:"entry"`
	Distance float64 `json:"distance"`
}

// VectorStore is a thread-safe in-memory vector index with JSON persistence.
type VectorStore struct {
	mu      sync.RWMutex
	entries map[string]Entry
	dim     int
}

// New creates a new VectorStore with the expected embedding dimension.
func New(dim int) *VectorStore {
	return &VectorStore{
		entries: make(map[string]Entry),
		dim:     dim,
	}
}

// Add inserts or updates a vector entry.
func (vs *VectorStore) Add(id string, vec []float32, metadata map[string]interface{}) {
	vs.mu.Lock()
	defer vs.mu.Unlock()
	vs.entries[id] = Entry{ID: id, Vector: vec, Metadata: metadata}
}

// Search returns the top-k most similar entries by cosine similarity.
func (vs *VectorStore) Search(query []float32, k int) []SearchResult {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	if len(vs.entries) == 0 {
		return nil
	}

	results := make([]SearchResult, 0, len(vs.entries))
	for _, entry := range vs.entries {
		dist := cosineSimilarity(query, entry.Vector)
		results = append(results, SearchResult{Entry: entry, Distance: dist})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance > results[j].Distance
	})

	if k < len(results) {
		results = results[:k]
	}
	return results
}

// Delete removes an entry by ID. Returns true if it existed.
func (vs *VectorStore) Delete(id string) bool {
	vs.mu.Lock()
	defer vs.mu.Unlock()
	_, ok := vs.entries[id]
	if ok {
		delete(vs.entries, id)
	}
	return ok
}

// Len returns the number of stored vectors.
func (vs *VectorStore) Len() int {
	vs.mu.RLock()
	defer vs.mu.RUnlock()
	return len(vs.entries)
}

// Save writes the store to a JSON file.
func (vs *VectorStore) Save(path string) error {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	data := struct {
		Entries map[string]Entry `json:"entries"`
		Dim     int              `json:"dim"`
	}{Entries: vs.entries, Dim: vs.dim}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(data)
}

// Load reads the store from a JSON file.
func Load(path string) (*VectorStore, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var data struct {
		Entries map[string]Entry `json:"entries"`
		Dim     int              `json:"dim"`
	}
	if err := json.NewDecoder(f).Decode(&data); err != nil {
		return nil, err
	}

	return &VectorStore{entries: data.Entries, dim: data.Dim}, nil
}

func cosineSimilarity(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		va := float64(a[i])
		vb := float64(b[i])
		dot += va * vb
		normA += va * va
		normB += vb * vb
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
