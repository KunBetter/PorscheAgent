package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"

	"github.com/kunbetter/porsche-agent/vectorstore/store"
)

var vs *store.VectorStore

func main() {
	dim := 1536 // DeepSeek / OpenAI text-embedding-3-small
	vs = store.New(dim)

	mux := http.NewServeMux()
	mux.HandleFunc("/health", handleHealth)
	mux.HandleFunc("/add", handleAdd)
	mux.HandleFunc("/search", handleSearch)
	mux.HandleFunc("/delete", handleDelete)
	mux.HandleFunc("/save", handleSave)
	mux.HandleFunc("/load", handleLoad)

	port := os.Getenv("VECTORSTORE_PORT")
	if port == "" {
		port = "9876"
	}

	// Auto-load from default path if it exists
	defaultPath := os.Getenv("VECTORSTORE_PATH")
	if defaultPath == "" {
		defaultPath = "porsche_memory.json"
	}
	if _, err := os.Stat(defaultPath); err == nil {
		loaded, err := store.Load(defaultPath)
		if err != nil {
			log.Printf("Warning: failed to load %s: %v", defaultPath, err)
		} else {
			vs = loaded
			log.Printf("Loaded %d vectors from %s", vs.Len(), defaultPath)
		}
	}

	log.Printf("Vector store listening on :%s (dim=%d)", port, dim)
	log.Fatal(http.ListenAndServe(":"+port, mux))
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status": "ok",
		"count":  vs.Len(),
	})
}

func handleAdd(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		ID       string                 `json:"id"`
		Vector   []float32              `json:"vector"`
		Metadata map[string]interface{} `json:"metadata"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.ID == "" {
		http.Error(w, "missing id", http.StatusBadRequest)
		return
	}
	if req.Metadata == nil {
		req.Metadata = make(map[string]interface{})
	}
	vs.Add(req.ID, req.Vector, req.Metadata)
	writeJSON(w, http.StatusCreated, map[string]string{"status": "ok", "id": req.ID})
}

func handleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Vector []float32 `json:"vector"`
		K      int       `json:"k"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.K <= 0 {
		req.K = 5
	}

	results := vs.Search(req.Vector, req.K)
	writeJSON(w, http.StatusOK, map[string]interface{}{"results": results})
}

func handleDelete(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}
	ok := vs.Delete(req.ID)
	if ok {
		writeJSON(w, http.StatusOK, map[string]string{"status": "deleted", "id": req.ID})
	} else {
		writeJSON(w, http.StatusNotFound, map[string]string{"status": "not found", "id": req.ID})
	}
}

func handleSave(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Path string `json:"path"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.Path == "" {
		req.Path = "porsche_memory.json"
	}
	if err := vs.Save(req.Path); err != nil {
		http.Error(w, "save failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "saved", "path": req.Path})
}

func handleLoad(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Path string `json:"path"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.Path == "" {
		req.Path = "porsche_memory.json"
	}
	loaded, err := store.Load(req.Path)
	if err != nil {
		http.Error(w, "load failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	vs = loaded
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status": "loaded",
		"path":   req.Path,
		"count":  vs.Len(),
	})
}

func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}
