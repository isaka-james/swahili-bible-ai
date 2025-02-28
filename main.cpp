#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <sstream>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <numeric>
#include <functional>
#include <iomanip>

// Structure to hold verse information (simplified)
struct Verse {
    std::string text;
    size_t id;  // Just a simple identifier
};

// Simple matrix class for transformer operations
class Matrix {
private:
    std::vector<float> data;
    size_t rows, cols;

public:
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols), data(rows * cols, 0.0f) {}
    
    Matrix(size_t rows, size_t cols, std::function<float()> initializer) 
        : rows(rows), cols(cols), data(rows * cols) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = initializer();
        }
    }

    float& at(size_t row, size_t col) {
        return data[row * cols + col];
    }
    
    float at(size_t row, size_t col) const {
        return data[row * cols + col];
    }
    
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

    // Matrix operations
    Matrix matmul(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::runtime_error("Incompatible dimensions for matrix multiplication");
        }
        
        Matrix result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < cols; ++k) {
                    sum += at(i, k) * other.at(k, j);
                }
                result.at(i, j) = sum;
            }
        }
        
        return result;
    }
    
    // Element-wise operations
    Matrix add(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::runtime_error("Incompatible dimensions for addition");
        }
        
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.at(i, j) = at(i, j) + other.at(i, j);
            }
        }
        
        return result;
    }
    
    // Apply a function to each element
    Matrix apply(std::function<float(float)> func) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.at(i, j) = func(at(i, j));
            }
        }
        
        return result;
    }
    
    // Softmax activation along rows
    Matrix softmax() const {
        Matrix result(rows, cols);
        
        for (size_t i = 0; i < rows; ++i) {
            // Find max for numerical stability
            float maxVal = at(i, 0);
            for (size_t j = 1; j < cols; ++j) {
                maxVal = std::max(maxVal, at(i, j));
            }
            
            // Calculate exp and sum
            float sum = 0.0f;
            for (size_t j = 0; j < cols; ++j) {
                float expVal = std::exp(at(i, j) - maxVal);
                result.at(i, j) = expVal;
                sum += expVal;
            }
            
            // Normalize
            for (size_t j = 0; j < cols; ++j) {
                result.at(i, j) /= sum;
            }
        }
        
        return result;
    }
    
    // Return the index of max value in each row
    std::vector<size_t> argmax() const {
        std::vector<size_t> indices(rows);
        
        for (size_t i = 0; i < rows; ++i) {
            size_t maxIdx = 0;
            float maxVal = at(i, 0);
            
            for (size_t j = 1; j < cols; ++j) {
                if (at(i, j) > maxVal) {
                    maxVal = at(i, j);
                    maxIdx = j;
                }
            }
            
            indices[i] = maxIdx;
        }
        
        return indices;
    }
};

// Simplified Transformer-like model for text generation
class TransformerModel {
private:
    // Vocabulary and embeddings
    std::unordered_map<std::string, size_t> word2idx;
    std::vector<std::string> idx2word;
    Matrix embeddings;
    
    // Simple attention mechanism
    Matrix queryWeight;
    Matrix keyWeight;
    Matrix valueWeight;
    
    // Output projection
    Matrix outputWeight;
    
    // Model hyperparameters
    size_t embeddingDim;
    size_t contextLen;
    size_t vocabularySize;
    
    // Random number generator
    std::mt19937 rng;
    
    // Convert text to token indices
    std::vector<size_t> tokenize(const std::string& text) {
        std::vector<size_t> tokens;
        std::istringstream iss(text);
        std::string word;
        
        while (iss >> word) {
            // Simple preprocessing: remove punctuation, convert to lowercase
            word.erase(std::remove_if(word.begin(), word.end(), 
                                     [](char c) { return std::ispunct(c); }), word.end());
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            
            if (!word.empty()) {
                if (word2idx.find(word) != word2idx.end()) {
                    tokens.push_back(word2idx[word]);
                } else {
                    tokens.push_back(word2idx["<unk>"]);  // Unknown token
                }
            }
        }
        
        return tokens;
    }
    
    // Convert token indices to input embeddings
    Matrix getInputEmbeddings(const std::vector<size_t>& tokens) {
        Matrix inputEmb(tokens.size(), embeddingDim);
        
        for (size_t i = 0; i < tokens.size(); ++i) {
            for (size_t j = 0; j < embeddingDim; ++j) {
                inputEmb.at(i, j) = embeddings.at(tokens[i], j);
            }
        }
        
        return inputEmb;
    }
    
    // Simple self-attention mechanism
    Matrix selfAttention(const Matrix& inputEmb) {
        // Q, K, V projections
        Matrix Q = inputEmb.matmul(queryWeight);
        Matrix K = inputEmb.matmul(keyWeight);
        Matrix V = inputEmb.matmul(valueWeight);
        
        // Calculate attention scores (Q * K^T / sqrt(d_k))
        Matrix scores(Q.getRows(), K.getRows());
        for (size_t i = 0; i < Q.getRows(); ++i) {
            for (size_t j = 0; j < K.getRows(); ++j) {
                float dot = 0.0f;
                for (size_t k = 0; k < Q.getCols(); ++k) {
                    dot += Q.at(i, k) * K.at(j, k);
                }
                scores.at(i, j) = dot / std::sqrt(static_cast<float>(Q.getCols()));
                
                // Add causal mask (if j > i, set score to very negative value)
                if (j > i) {
                    scores.at(i, j) = -1e9;
                }
            }
        }
        
        // Apply softmax to get attention weights
        Matrix attentionWeights = scores.softmax();
        
        // Calculate weighted sum of values
        Matrix output(Q.getRows(), V.getCols());
        for (size_t i = 0; i < output.getRows(); ++i) {
            for (size_t j = 0; j < output.getCols(); ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < V.getRows(); ++k) {
                    sum += attentionWeights.at(i, k) * V.at(k, j);
                }
                output.at(i, j) = sum;
            }
        }
        
        return output;
    }
    
    // Forward pass
    Matrix forward(const std::vector<size_t>& tokens) {
        // Get input embeddings
        Matrix inputEmb = getInputEmbeddings(tokens);
        
        // Apply self-attention
        Matrix attentionOutput = selfAttention(inputEmb);
        
        // Project to vocabulary size
        Matrix logits = attentionOutput.matmul(outputWeight);
        
        return logits;
    }
    
    // Sample from distribution
    size_t sample(const std::vector<float>& probs) {
        std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
        return dist(rng);
    }

public:
    TransformerModel(size_t embeddingDim = 64, size_t contextLen = 16) 
        : embeddingDim(embeddingDim), 
          contextLen(contextLen),
          rng(std::chrono::steady_clock::now().time_since_epoch().count()),
          embeddings(0, 0),
          queryWeight(0, 0),
          keyWeight(0, 0),
          valueWeight(0, 0),
          outputWeight(0, 0) {
        
        // Initialize vocabulary with special tokens
        word2idx["<pad>"] = 0;
        word2idx["<unk>"] = 1;
        idx2word = {"<pad>", "<unk>"};
        vocabularySize = 2;  // Will be updated during build
    }
    
    // Build vocabulary and initialize model from corpus
    void buildModel(const std::vector<Verse>& verses) {
        std::cout << "Building transformer model...\n";
        
        // Step 1: Build vocabulary
        std::map<std::string, int> wordCounts;
        
        for (const auto& verse : verses) {
            std::istringstream iss(verse.text);
            std::string word;
            
            while (iss >> word) {
                // Simple preprocessing
                word.erase(std::remove_if(word.begin(), word.end(), 
                                         [](char c) { return std::ispunct(c); }), word.end());
                std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                
                if (!word.empty()) {
                    wordCounts[word]++;
                }
            }
        }
        
        // Sort words by frequency
        std::vector<std::pair<std::string, int>> wordFreq;
        for (const auto& pair : wordCounts) {
            wordFreq.emplace_back(pair);
        }
        
        std::sort(wordFreq.begin(), wordFreq.end(),
                 [](const auto& a, const auto& b) {
                     return a.second > b.second;
                 });
        
        // Take top N words to limit vocabulary size
        const size_t MAX_VOCAB_SIZE = 5000;
        size_t vocabLimit = std::min(MAX_VOCAB_SIZE, wordFreq.size());
        
        // Add words to vocabulary
        for (size_t i = 0; i < vocabLimit; ++i) {
            const std::string& word = wordFreq[i].first;
            
            if (word2idx.find(word) == word2idx.end()) {
                size_t idx = idx2word.size();
                word2idx[word] = idx;
                idx2word.push_back(word);
            }
        }
        
        vocabularySize = idx2word.size();
        std::cout << "Vocabulary size: " << vocabularySize << std::endl;
        
        // Print top words
        std::cout << "Top 20 most frequent words in the Bible:\n";
        for (size_t i = 0; i < 20 && i < wordFreq.size(); ++i) {
            std::cout << std::setw(15) << std::left << wordFreq[i].first 
                      << ": " << wordFreq[i].second << " occurrences\n";
        }
        
        // Step 2: Initialize model parameters
        // Initialize random number generator with normal distribution
        std::normal_distribution<float> norm(0.0f, 0.02f);
        auto normalInit = [&]() { return norm(rng); };
        
        // Initialize embeddings
        embeddings = Matrix(vocabularySize, embeddingDim, normalInit);
        
        // Initialize attention weights
        queryWeight = Matrix(embeddingDim, embeddingDim, normalInit);
        keyWeight = Matrix(embeddingDim, embeddingDim, normalInit);
        valueWeight = Matrix(embeddingDim, embeddingDim, normalInit);
        
        // Initialize output projection
        outputWeight = Matrix(embeddingDim, vocabularySize, normalInit);
        
        std::cout << "Transformer model initialized with embedding dimension " 
                  << embeddingDim << " and context length " << contextLen << std::endl;
    }
    
    // Generate text using the model
    std::string generateText(const std::string& prompt, size_t maxLength = 50, float temperature = 0.8f) {
        // Tokenize prompt
        std::vector<size_t> tokens = tokenize(prompt);
        
        // Ensure tokens are within context length
        if (tokens.size() > contextLen) {
            tokens.erase(tokens.begin(), tokens.begin() + (tokens.size() - contextLen));
        }
        
        std::string result = prompt;
        
        for (size_t i = 0; i < maxLength; ++i) {
            // Forward pass to get next token probabilities
            Matrix logits = forward(tokens);
            
            // Get probabilities for the next token (last position in sequence)
            size_t lastPos = tokens.size() - 1;
            std::vector<float> probs(vocabularySize);
            
            // Apply temperature scaling
            for (size_t j = 0; j < vocabularySize; ++j) {
                probs[j] = std::exp(logits.at(lastPos, j) / temperature);
            }
            
            // Normalize probabilities
            float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
            for (size_t j = 0; j < vocabularySize; ++j) {
                probs[j] /= sum;
            }
            
            // Sample next token
            size_t nextToken = sample(probs);
            
            // Add to result
            result += " " + idx2word[nextToken];
            
            // Add to current tokens and maintain context length
            tokens.push_back(nextToken);
            if (tokens.size() > contextLen) {
                tokens.erase(tokens.begin());
            }
        }
        
        return result;
    }
    
    // Get top related words based on embedding similarity
    std::vector<std::pair<std::string, float>> getRelatedWords(const std::string& word, int n = 10) {
        std::vector<std::pair<std::string, float>> result;
        
        // Check if word is in vocabulary
        if (word2idx.find(word) == word2idx.end()) {
            return result;
        }
        
        size_t wordIdx = word2idx[word];
        std::vector<float> wordEmbedding(embeddingDim);
        
        for (size_t i = 0; i < embeddingDim; ++i) {
            wordEmbedding[i] = embeddings.at(wordIdx, i);
        }
        
        // Calculate cosine similarity with all other words
        std::vector<std::pair<std::string, float>> similarities;
        
        for (size_t i = 0; i < vocabularySize; ++i) {
            if (i == wordIdx) continue;
            
            // Calculate dot product
            float dotProduct = 0.0f;
            for (size_t j = 0; j < embeddingDim; ++j) {
                dotProduct += wordEmbedding[j] * embeddings.at(i, j);
            }
            
            // Calculate magnitudes
            float mag1 = 0.0f, mag2 = 0.0f;
            for (size_t j = 0; j < embeddingDim; ++j) {
                mag1 += wordEmbedding[j] * wordEmbedding[j];
                mag2 += embeddings.at(i, j) * embeddings.at(i, j);
            }
            
            mag1 = std::sqrt(mag1);
            mag2 = std::sqrt(mag2);
            
            // Calculate similarity
            float similarity = dotProduct / (mag1 * mag2);
            
            similarities.emplace_back(idx2word[i], similarity);
        }
        
        // Sort by similarity (descending)
        std::sort(similarities.begin(), similarities.end(),
                 [](const auto& a, const auto& b) {
                     return a.second > b.second;
                 });
        
        // Return top n
        for (int i = 0; i < n && i < static_cast<int>(similarities.size()); ++i) {
            result.push_back(similarities[i]);
        }
        
        return result;
    }
    
    // Get top words (for statistics)
    std::vector<std::pair<std::string, int>> getTopWords(const std::vector<Verse>& verses, int n = 10) {
        std::map<std::string, int> wordCounts;
        
        for (const auto& verse : verses) {
            std::istringstream iss(verse.text);
            std::string word;
            
            while (iss >> word) {
                // Simple preprocessing
                word.erase(std::remove_if(word.begin(), word.end(), 
                                         [](char c) { return std::ispunct(c); }), word.end());
                std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                
                if (!word.empty()) {
                    wordCounts[word]++;
                }
            }
        }
        
        // Convert to vector for sorting
        std::vector<std::pair<std::string, int>> wordFreq(wordCounts.begin(), wordCounts.end());
        
        // Sort by frequency
        std::sort(wordFreq.begin(), wordFreq.end(),
                 [](const auto& a, const auto& b) {
                     return a.second > b.second;
                 });
        
        // Return top n
        std::vector<std::pair<std::string, int>> result;
        for (int i = 0; i < n && i < static_cast<int>(wordFreq.size()); ++i) {
            result.push_back(wordFreq[i]);
        }
        
        return result;
    }
};

// Main Bible text analysis class
class BibleTextAnalyzer {
private:
    std::vector<Verse> verses;
    std::shared_ptr<TransformerModel> model;
    std::mt19937 rng;

public:
    BibleTextAnalyzer() : rng(std::chrono::steady_clock::now().time_since_epoch().count()) {
        // Initialize transformer model
        model = std::make_shared<TransformerModel>(128, 24);  // 128-dim embeddings, context length 24
    }

    // Load verses from file - simplified to just read lines as verses
    bool loadBibleFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return false;
        }

        std::string line;
        size_t id = 0;
        while (std::getline(file, line)) {
            // Skip empty lines
            if (line.empty()) continue;
            
            // Store each line as a verse
            Verse verse{line, id++};
            verses.push_back(verse);
        }

        std::cout << "Loaded " << verses.size() << " verses from the Bible." << std::endl;
        return !verses.empty();
    }

    // Build the transformer model
    void buildModel() {
        model->buildModel(verses);
    }

    // Generate text using the model
    std::string generateText(const std::string& prompt, size_t maxLength = 50, float temperature = 0.8f) {
        return model->generateText(prompt, maxLength, temperature);
    }

    // Find verses containing a specific word or phrase
    std::vector<Verse> findVerses(const std::string& searchTerm) {
        std::vector<Verse> matchingVerses;
        std::string lowerSearchTerm = searchTerm;
        std::transform(lowerSearchTerm.begin(), lowerSearchTerm.end(), lowerSearchTerm.begin(), ::tolower);

        for (const auto& verse : verses) {
            std::string lowerText = verse.text;
            std::transform(lowerText.begin(), lowerText.end(), lowerText.begin(), ::tolower);
            
            if (lowerText.find(lowerSearchTerm) != std::string::npos) {
                matchingVerses.push_back(verse);
            }
        }

        return matchingVerses;
    }

    // Get related words to a given word from the model
    void showRelatedWords(const std::string& word, int count = 10) {
        auto relatedWords = model->getRelatedWords(word, count);
        
        std::cout << "\nWords related to '" << word << "':\n";
        for (const auto& pair : relatedWords) {
            std::cout << std::setw(15) << std::left << pair.first 
                      << " (similarity: " << std::fixed << std::setprecision(4) << pair.second << ")\n";
        }
    }

    // Get statistics about the Bible text
    void printStatistics() {
        // Count total words
        size_t totalWords = 0;
        std::map<std::string, int> wordCounts;
        
        for (const auto& verse : verses) {
            std::istringstream iss(verse.text);
            std::string word;
            
            while (iss >> word) {
                // Simple preprocessing
                word.erase(std::remove_if(word.begin(), word.end(), 
                                         [](char c) { return std::ispunct(c); }), word.end());
                std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                
                if (!word.empty()) {
                    wordCounts[word]++;
                    totalWords++;
                }
            }
        }
        
        // Calculate average verse length
        float avgVerseLength = static_cast<float>(totalWords) / verses.size();
        
        // Get distribution of verse lengths
        std::vector<size_t> verseLengths;
        for (const auto& verse : verses) {
            std::istringstream iss(verse.text);
            std::string word;
            size_t length = 0;
            
            while (iss >> word) {
                length++;
            }
            
            verseLengths.push_back(length);
        }
        
        // Sort for percentiles
        std::sort(verseLengths.begin(), verseLengths.end());
        
        size_t p25 = verseLengths[verseLengths.size() * 0.25];
        size_t p50 = verseLengths[verseLengths.size() * 0.5];
        size_t p75 = verseLengths[verseLengths.size() * 0.75];
        
        // Print statistics
        std::cout << "\n=== Bible Statistics ===\n";
        std::cout << "Total verses: " << verses.size() << std::endl;
        std::cout << "Total words: " << totalWords << std::endl;
        std::cout << "Unique words: " << wordCounts.size() << std::endl;
        std::cout << "Average words per verse: " << std::fixed << std::setprecision(1) << avgVerseLength << std::endl;
        std::cout << "Verse length distribution: 25th percentile: " << p25 
                  << ", Median: " << p50 
                  << ", 75th percentile: " << p75 << std::endl;
        
        // Print top words
        std::cout << "\nTop 10 most frequent words:\n";
        auto topWords = model->getTopWords(verses, 10);
        for (const auto& pair : topWords) {
            std::cout << std::setw(15) << std::left << pair.first 
                      << ": " << pair.second << " occurrences\n";
        }
        
        // Print some example verses
        std::cout << "\nSample verses:\n";
        std::uniform_int_distribution<size_t> dist(0, verses.size() - 1);
        for (int i = 0; i < 3; ++i) {
            size_t idx = dist(rng);
            std::cout << "Verse #" << idx << ": " << verses[idx].text << std::endl;
        }
    }
};

int main() {
    BibleTextAnalyzer analyzer;
    
    std::cout << "Bible Text Analyzer using Transformer Model\n";
    std::cout << "-------------------------------------------\n";
    
    // Load Bible verses from file
    if (!analyzer.loadBibleFromFile("bible.txt")) {
        std::cerr << "Failed to load Bible text. Make sure 'bible.txt' exists in the current directory." << std::endl;
        return 1;
    }
    
    // Build the transformer model
    analyzer.buildModel();
    
    // Print statistics
    analyzer.printStatistics();
    
    // Generate some text
    std::cout << "\n=== Generated Bible-like text ===\n";
    std::cout << "Prompt: 'In the beginning'\n";
    std::cout << analyzer.generateText("In the beginning", 50, 0.7f) << std::endl;
    
    std::cout << "\nPrompt: 'Love thy'\n";
    std::cout << analyzer.generateText("Love thy", 50, 0.7f) << std::endl;
    
    // Show related words
    analyzer.showRelatedWords("love");
    analyzer.showRelatedWords("faith");
    analyzer.showRelatedWords("sin");
    
    // Search for verses
    std::string searchTerm = "love thy neighbor";
    auto results = analyzer.findVerses(searchTerm);
    
    std::cout << "\n=== Search results for '" << searchTerm << "' ===\n";
    std::cout << "Found " << results.size() << " matching verses:\n";
    for (size_t i = 0; i < std::min(results.size(), size_t(5)); ++i) {
        const auto& verse = results[i];
        std::cout << "Verse #" << verse.id << ": " << verse.text << std::endl;
    }
    
    if (results.size() > 5) {
        std::cout << "... and " << (results.size() - 5) << " more results.\n";
    }
    
    // Interactive mode
    std::cout << "\n=== Interactive Mode ===\n";
    std::cout << "Enter a prompt to generate text or 'q' to quit:\n";
    
    std::string input;
    while (true) {
        std::cout << "> ";
        std::getline(std::cin, input);
        
        if (input == "q" || input == "quit" || input == "exit") {
            break;
        }
        
        if (!input.empty()) {
            std::cout << analyzer.generateText(input, 50, 0.7f) << "\n\n";
        }
    }
    
    return 0;
}