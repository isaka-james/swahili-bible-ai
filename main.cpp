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
#include <iterator>

// Structure to hold verse information (simplified)
struct Verse
{
    std::string text;
    size_t id; // Just a simple identifier
};

// Simple matrix class for transformer operations
class Matrix
{
private:
    std::vector<float> data;
    size_t rows, cols;

public:
    Matrix(size_t rows, size_t cols)
        : rows(rows), cols(cols), data(rows * cols, 0.0f) {}

    Matrix(size_t rows, size_t cols, std::function<float()> initializer)
        : rows(rows), cols(cols), data(rows * cols)
    {
        for (size_t i = 0; i < data.size(); ++i)
        {
            data[i] = initializer();
        }
    }

    // Element access
    float &at(size_t row, size_t col) { return data[row * cols + col]; }
    float at(size_t row, size_t col) const { return data[row * cols + col]; }

    // Dimensions
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

    // Matrix multiplication
    Matrix matmul(const Matrix &other) const
    {
        if (cols != other.rows)
        {
            throw std::runtime_error("Matmul dimension mismatch");
        }

        Matrix result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < other.cols; ++j)
            {
                float sum = 0.0f;
                for (size_t k = 0; k < cols; ++k)
                {
                    sum += at(i, k) * other.at(k, j);
                }
                result.at(i, j) = sum;
            }
        }
        return result;
    }

    // Matrix addition
    Matrix add(const Matrix &other) const
    {
        if (rows != other.rows || cols != other.cols)
        {
            throw std::runtime_error("Matrix addition dimension mismatch");
        }

        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                result.at(i, j) = at(i, j) + other.at(i, j);
            }
        }
        return result;
    }

    // Element-wise operation
    Matrix apply(std::function<float(float)> func) const
    {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                result.at(i, j) = func(at(i, j));
            }
        }
        return result;
    }

    // Column slicing
    Matrix slice(size_t start_col, size_t end_col) const
    {
        if (end_col > cols || start_col >= end_col)
        {
            throw std::runtime_error("Invalid slice dimensions");
        }

        Matrix result(rows, end_col - start_col);
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = start_col; j < end_col; ++j)
            {
                result.at(i, j - start_col) = at(i, j);
            }
        }
        return result;
    }

    // Softmax activation
    Matrix softmax() const
    {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i)
        {
            float max_val = *std::max_element(&data[i * cols], &data[i * cols] + cols);
            float sum = 0.0f;

            for (size_t j = 0; j < cols; ++j)
            {
                result.at(i, j) = std::exp(at(i, j) - max_val);
                sum += result.at(i, j);
            }

            for (size_t j = 0; j < cols; ++j)
            {
                result.at(i, j) /= sum;
            }
        }
        return result;
    }

    // Argmax for each row
    std::vector<size_t> argmax() const
    {
        std::vector<size_t> indices(rows);
        for (size_t i = 0; i < rows; ++i)
        {
            indices[i] = static_cast<size_t>(std::max_element(
                                                 &data[i * cols], &data[i * cols] + cols) -
                                             &data[i * cols]);
        }
        return indices;
    }
};

class TransformerModel
{
private:
    // Vocabulary and embeddings
    std::unordered_map<std::string, size_t> word2idx;
    std::vector<std::string> idx2word;
    Matrix embeddings;

    // Model parameters
    Matrix queryWeight;
    Matrix keyWeight;
    Matrix valueWeight;
    Matrix outputWeight;
    Matrix ffn1Weight;
    Matrix ffn2Weight;

    // Hyperparameters
    size_t embeddingDim;
    size_t contextLen;
    size_t vocabularySize;
    int numLayers;
    int numHeads;

    // Random number generator
    std::mt19937 rng;

    // ================== Model Components ==================
    Matrix calculateAttention(const Matrix &Q, const Matrix &K, const Matrix &V) const
    {
        Matrix scores(Q.getRows(), K.getRows());
        for (size_t i = 0; i < Q.getRows(); ++i)
        {
            for (size_t j = 0; j < K.getRows(); ++j)
            {
                float dot = 0.0f;
                for (size_t k = 0; k < Q.getCols(); ++k)
                {
                    dot += Q.at(i, k) * K.at(j, k);
                }
                scores.at(i, j) = dot / std::sqrt(static_cast<float>(Q.getCols()));
            }
        }
        Matrix attention = scores.softmax();
        return attention.matmul(V);
    }

    Matrix layerNorm(const Matrix &input) const
    {
        Matrix result(input.getRows(), input.getCols());
        for (size_t i = 0; i < input.getRows(); ++i)
        {
            float mean = 0.0f, var = 0.0f;
            for (size_t j = 0; j < input.getCols(); ++j)
            {
                mean += input.at(i, j);
            }
            mean /= input.getCols();

            for (size_t j = 0; j < input.getCols(); ++j)
            {
                var += std::pow(input.at(i, j) - mean, 2);
            }
            var /= input.getCols();

            const float epsilon = 1e-5f;
            for (size_t j = 0; j < input.getCols(); ++j)
            {
                result.at(i, j) = (input.at(i, j) - mean) / std::sqrt(var + epsilon);
            }
        }
        return result;
    }

    Matrix feedForward(const Matrix &input) const
    {
        Matrix hidden = input.matmul(ffn1Weight);
        hidden = hidden.apply([](float x)
                              { return std::max(0.0f, x); }); // ReLU
        return hidden.matmul(ffn2Weight);
    }

    Matrix multiHeadAttention(const Matrix &inputEmb) const
    {
        const size_t headDim = embeddingDim / numHeads;
        Matrix result(inputEmb.getRows(), embeddingDim);

        for (int h = 0; h < numHeads; ++h)
        {
            // Slice weights for this head
            Matrix Q = inputEmb.matmul(queryWeight.slice(h * headDim, (h + 1) * headDim));
            Matrix K = inputEmb.matmul(keyWeight.slice(h * headDim, (h + 1) * headDim));
            Matrix V = inputEmb.matmul(valueWeight.slice(h * headDim, (h + 1) * headDim));

            Matrix headOutput = calculateAttention(Q, K, V);

            // Concatenate head outputs
            for (size_t i = 0; i < inputEmb.getRows(); ++i)
            {
                for (size_t j = 0; j < headDim; ++j)
                {
                    result.at(i, h * headDim + j) = headOutput.at(i, j);
                }
            }
        }
        return result;
    }

    // ================== Core Methods ==================
    std::vector<size_t> tokenize(const std::string &text)
    {
        std::vector<size_t> tokens;
        std::istringstream iss(text);
        std::string word;

        while (iss >> word)
        {
            // Preprocess
            word.erase(std::remove_if(word.begin(), word.end(),
                                      [](char c)
                                      { return std::ispunct(c); }),
                       word.end());
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);

            if (word.empty())
                continue;

            // Handle OOV with subword tokenization
            if (word2idx.count(word))
            {
                tokens.push_back(word2idx[word]);
            }
            else
            {
                bool found = false;
                for (size_t i = 1; i < word.size(); ++i)
                {
                    std::string prefix = word.substr(0, i);
                    std::string suffix = word.substr(i);
                    if (word2idx.count(prefix) && word2idx.count(suffix))
                    {
                        tokens.push_back(word2idx[prefix]);
                        tokens.push_back(word2idx[suffix]);
                        found = true;
                        break;
                    }
                }
                if (!found)
                    tokens.push_back(word2idx["<unk>"]);
            }
        }
        return tokens;
    }

    Matrix getInputEmbeddings(const std::vector<size_t> &tokens) const
    {
        Matrix inputEmb(tokens.size(), embeddingDim);

        // Word embeddings
        for (size_t i = 0; i < tokens.size(); ++i)
        {
            if (tokens[i] >= vocabularySize)
                continue;
            for (size_t j = 0; j < embeddingDim; ++j)
            {
                inputEmb.at(i, j) = embeddings.at(tokens[i], j);
            }
        }

        // Positional encoding
        for (size_t i = 0; i < tokens.size(); ++i)
        {
            for (size_t j = 0; j < embeddingDim; ++j)
            {
                float angle = i / std::pow(10000.0f, (2.0f * (j / 2)) / embeddingDim);
                if (j % 2 == 0)
                {
                    inputEmb.at(i, j) += std::sin(angle);
                }
                else
                {
                    inputEmb.at(i, j) += std::cos(angle);
                }
            }
        }
        return inputEmb;
    }

public:
    // ================== Public Interface ==================
    TransformerModel(size_t embeddingDim = 256, size_t contextLen = 32,
                     int numLayers = 4, int numHeads = 8)
        : embeddingDim(embeddingDim),
          contextLen(contextLen),
          numLayers(numLayers),
          numHeads(numHeads),
          queryWeight(embeddingDim, embeddingDim),
          keyWeight(embeddingDim, embeddingDim),
          valueWeight(embeddingDim, embeddingDim),
          outputWeight(embeddingDim, 1), // Temporary size, updated in buildModel
          ffn1Weight(embeddingDim, 4 * embeddingDim),
          ffn2Weight(4 * embeddingDim, embeddingDim),
          embeddings(0, 0),
          rng(std::chrono::steady_clock::now().time_since_epoch().count())
    {
        word2idx["<pad>"] = 0;
        word2idx["<unk>"] = 1;
        idx2word = {"<pad>", "<unk>"};
        vocabularySize = 2;
    }

    void buildModel(const std::vector<Verse> &verses)
    {
        // Vocabulary construction
        std::map<std::string, int> wordCounts;
        for (const auto &verse : verses)
        {
            std::istringstream iss(verse.text);
            std::string word;
            while (iss >> word)
            {
                word.erase(std::remove_if(word.begin(), word.end(), 
                          [](char c) {
                    return std::ispunct(c); 
                }), word.end());
                std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                if (!word.empty()) wordCounts[word]++;
            }
        }

        // Sort by frequency
        std::vector<std::pair<std::string, int>> wordFreq(wordCounts.begin(), wordCounts.end());
        std::sort(wordFreq.begin(), wordFreq.end(),
                  [](auto &a, auto &b)
                  { return a.second > b.second; });

        // Build vocabulary
        const size_t maxVocab = 5000;
        vocabularySize = std::min(wordFreq.size(), maxVocab) + 2; // +2 for special tokens
        idx2word.resize(vocabularySize);
        for (size_t i = 0; i < vocabularySize - 2; ++i)
        {
            word2idx[wordFreq[i].first] = i + 2;
            idx2word[i + 2] = wordFreq[i].first;
        }

        // Initialize parameters
        std::normal_distribution<float> dist(0.0f, 0.02f);
        auto init = [&]
        { return dist(rng); };

        embeddings = Matrix(vocabularySize, embeddingDim, init);
        queryWeight = Matrix(embeddingDim, embeddingDim, init);
        keyWeight = Matrix(embeddingDim, embeddingDim, init);
        valueWeight = Matrix(embeddingDim, embeddingDim, init);
        outputWeight = Matrix(embeddingDim, vocabularySize, init);
        ffn1Weight = Matrix(embeddingDim, 4 * embeddingDim, init);
        ffn2Weight = Matrix(4 * embeddingDim, embeddingDim, init);
    }

    Matrix forward(const std::vector<size_t> &tokens) const
    {
        if (tokens.empty())
            return Matrix(0, 0);

        Matrix x = getInputEmbeddings(tokens);

        for (int layer = 0; layer < numLayers; ++layer)
        {
            // Self-attention
            Matrix attn = multiHeadAttention(x);
            x = x.add(attn);
            x = layerNorm(x);

            // Feed-forward
            Matrix ff = feedForward(x);
            x = x.add(ff);
            x = layerNorm(x);
        }

        return x.matmul(outputWeight);
    }

    std::string generateText(const std::string &prompt, size_t maxLen = 50,
                             float temp = 1.0f, int topK = 40)
    {
        std::vector<size_t> tokens = tokenize(prompt);
        if (tokens.empty())
            tokens.push_back(word2idx["<unk>"]);

        std::string result = prompt;
        for (size_t i = 0; i < maxLen; ++i)
        {
            // Truncate to context length
            if (tokens.size() > contextLen)
            {
                tokens.erase(tokens.begin(), tokens.end() - contextLen);
            }

            Matrix logits = forward(tokens);
            size_t lastPos = tokens.size() - 1;

            // Get probabilities
            std::vector<float> probs(vocabularySize);
            for (size_t j = 0; j < vocabularySize; ++j)
            {
                probs[j] = std::exp(logits.at(lastPos, j) / temp);
            }

            // Normalize
            float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
            for (auto &p : probs)
                p /= sum;

            // Sample
            size_t nextToken = sample(probs, topK);
            std::string word = idx2word[nextToken];
            result += " " + word;
            tokens.push_back(nextToken);
        }
        return result;
    }

    size_t sample(const std::vector<float> &probs, int topK) {
        std::vector<size_t> indices(probs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + topK, indices.end(),
                          [&probs](size_t a, size_t b) { return probs[a] > probs[b]; });
    
        std::vector<float> topProbs(topK);
        float sum = 0.0f;
        for (int i = 0; i < topK; ++i) {
            topProbs[i] = probs[indices[i]];
            sum += topProbs[i];
        }
    
        std::uniform_real_distribution<float> dist(0.0f, sum);
        float threshold = dist(rng);  // Remove const from rng
        float accum = 0.0f;
        for (int i = 0; i < topK; ++i) {
            accum += topProbs[i];
            if (accum >= threshold) {
                return indices[i];
            }
        }
        return indices.back();
    }

    // ================== Utility Methods ==================
    std::vector<std::pair<std::string, float>> getRelatedWords(const std::string &word, int topN = 10) const
    {
        std::vector<std::pair<std::string, float>> results;
        if (!word2idx.count(word))
            return results;

        const size_t wordIdx = word2idx.at(word);
        Matrix wordVec = embeddings.slice(wordIdx, wordIdx + 1);

        std::vector<std::pair<float, std::string>> similarities;
        for (size_t i = 0; i < vocabularySize; ++i)
        {
            if (i == wordIdx)
                continue;

            Matrix currVec = embeddings.slice(i, i + 1);
            float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
            for (size_t j = 0; j < embeddingDim; ++j)
            {
                dot += wordVec.at(0, j) * currVec.at(0, j);
                norm1 += wordVec.at(0, j) * wordVec.at(0, j);
                norm2 += currVec.at(0, j) * currVec.at(0, j);
            }
            float cosine = dot / (std::sqrt(norm1) * std::sqrt(norm2));
            similarities.emplace_back(cosine, idx2word[i]);
        }

        std::sort(similarities.rbegin(), similarities.rend());
        for (int i = 0; i < topN && i < similarities.size(); ++i)
        {
            results.emplace_back(similarities[i].second, similarities[i].first);
        }
        return results;
    }
};

class BibleTextAnalyzer
{
private:
    std::vector<Verse> verses;
    std::shared_ptr<TransformerModel> model;
    mutable std::mt19937 rng;

public:
    BibleTextAnalyzer() : rng(std::chrono::steady_clock::now().time_since_epoch().count())
    {
        model = std::make_shared<TransformerModel>(256, 64, 4, 8); // Embedding dim 256, context 64, 4 layers, 8 heads
    }

    bool loadBibleFromFile(const std::string &filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return false;
        }

        verses.clear();
        std::string line;
        size_t id = 0;
        while (std::getline(file, line))
        {
            if (!line.empty())
            {
                verses.push_back({line, id++});
            }
        }

        std::cout << "Loaded " << verses.size() << " verses.\n";
        return !verses.empty();
    }

    void buildModel()
    {
        if (verses.empty())
        {
            std::cerr << "Error: No verses loaded to build model\n";
            return;
        }
        model->buildModel(verses);
    }

    std::string generateText(const std::string &prompt, size_t maxLength = 50, float temperature = 0.8f)
    {
        if (verses.empty())
        {
            return "Error: No Bible text loaded";
        }
        return model->generateText(prompt, maxLength, temperature);
    }

    std::vector<Verse> findVerses(const std::string &searchTerm) const
    {
        std::vector<Verse> results;
        std::string lowerTerm = searchTerm;
        std::transform(lowerTerm.begin(), lowerTerm.end(), lowerTerm.begin(), ::tolower);

        for (const auto &verse : verses)
        {
            std::string text = verse.text;
            std::transform(text.begin(), text.end(), text.begin(), ::tolower);
            if (text.find(lowerTerm) != std::string::npos)
            {
                results.push_back(verse);
            }
        }
        return results;
    }

    void showRelatedWords(const std::string &word, int count = 10) const
    {
        auto related = model->getRelatedWords(word, count);
        std::cout << "\nWords related to '" << word << "':\n";
        for (const auto &[w, sim] : related)
        {
            std::cout << std::setw(15) << std::left << w
                      << " (similarity: " << std::fixed << std::setprecision(3) << sim << ")\n";
        }
    }

    void printStatistics() const {
        if (verses.empty()) {
            std::cout << "No data loaded\n";
            return;
        }
    
        // Word statistics
        size_t totalWords = 0;
        std::unordered_map<std::string, int> wordCounts;
    
        for (const auto &verse : verses) {
            std::istringstream iss(verse.text);
            std::string word;
            while (iss >> word) {
                // Match model's preprocessing
                word.erase(std::remove_if(word.begin(), word.end(),
                                          [](char c) { return std::ispunct(c); }),
                           word.end());
                std::transform(word.begin(), word.end(), word.begin(), ::tolower);
    
                if (!word.empty()) {
                    wordCounts[word]++;
                    totalWords++;
                }
            }
        }
    
        // Verse length statistics
        std::vector<size_t> lengths;
        for (const auto &verse : verses) {
            std::istringstream iss(verse.text);
            lengths.push_back(std::distance(std::istream_iterator<std::string>(iss),
                                            std::istream_iterator<std::string>()));
        }
    
        // Calculate percentiles
        std::sort(lengths.begin(), lengths.end());
        size_t p25 = lengths.size() > 0 ? lengths[lengths.size() / 4] : 0;
        size_t p50 = lengths.size() > 0 ? lengths[lengths.size() / 2] : 0;
        size_t p75 = lengths.size() > 0 ? lengths[3 * lengths.size() / 4] : 0;
    
        // Print report
        std::cout << "\n=== Bible Statistics ===\n"
                  << "Total verses: " << verses.size() << "\n"
                  << "Total words: " << totalWords << "\n"
                  << "Unique words: " << wordCounts.size() << "\n"
                  << "Average words/verse: " << std::fixed << std::setprecision(1)
                  << (totalWords / (double)verses.size()) << "\n"
                  << "Verse length percentiles - 25th: " << p25
                  << ", 50th: " << p50
                  << ", 75th: " << p75 << "\n";
    
        // Show sample verses
        std::uniform_int_distribution<size_t> dist(0, verses.size() - 1);
        for (int i = 0; i < 3; i++) {
            const auto &v = verses[dist(rng)];  // Remove const from rng
            std::cout << "[" << v.id << "] " << v.text << "\n";
        }
    }
};

int main() {
    BibleTextAnalyzer analyzer;
    
    std::cout << "Bible Text Analyzer using Transformer Model\n"
              << "-------------------------------------------\n";

    // Load Bible text
    if (!analyzer.loadBibleFromFile("bible.txt")) {
        std::cerr << "\nError: Failed to load Bible text. "
                  << "Ensure 'bible.txt' exists in the current directory.\n";
        return 1;
    }

    // Build and validate model
    try {
        std::cout << "\nBuilding language model...\n";
        analyzer.buildModel();
    } catch (const std::exception& e) {
        std::cerr << "\nModel build failed: " << e.what() << "\n";
        return 2;
    }

    // Main program flow
    analyzer.printStatistics();

    // Text generation examples
    std::cout << "\n=== Generated Bible-like Text ===\n";
    std::cout << "Prompt: 'Yesu ni'\n";
    std::cout << analyzer.generateText("Yesu ni", 50, 0.7f) << "\n\n";

    std::cout << "Prompt: 'Mungu ni '\n";
    std::cout << analyzer.generateText("Mungu ni", 50, 0.7f) << "\n\n";

    // Word relationships
    analyzer.showRelatedWords("Yesu");
    analyzer.showRelatedWords("Shetani");
    analyzer.showRelatedWords("Mbingu");

    // Verse search
    std::string searchTerm = "Hapo mwanzo";
    auto results = analyzer.findVerses(searchTerm);
    
    std::cout << "\n=== Search Results for '" << searchTerm << "' ===\n"
              << "Found " << results.size() << " matching verses:\n";
              
    for (size_t i = 0; i < std::min(results.size(), size_t(5)); ++i) {
        std::cout << "Verse #" << results[i].id << ": " 
                  << results[i].text << "\n";
    }
    if (results.size() > 5) {
        std::cout << "... and " << (results.size() - 5) 
                  << " more results.\n";
    }

    // Interactive mode
    std::cout << "\n=== Interactive Mode ===\n"
              << "Enter prompts for text generation (or 'quit' to exit)\n";
    
    while (true) {
        std::cout << "\n> ";
        std::string input;
        std::getline(std::cin, input);
        
        if (input.empty()) continue;
        if (input == "q" || input == "quit" || input == "exit") break;

        try {
            std::cout << "\nGenerated text:\n" 
                      << analyzer.generateText(input, 50, 0.7f) 
                      << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Generation error: " << e.what() << "\n";
        }
    }

    std::cout << "\nExiting Bible Analyzer. Peace be with you!\n";
    return 0;
}