#include <eigen3/Eigen/Dense>
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

class TransformerModel
{
private:
    // Vocabulary and embeddings
    std::unordered_map<std::string, size_t> word2idx;
    std::vector<std::string> idx2word;
    Eigen::MatrixXf embeddings;

    // Model parameters
    Eigen::MatrixXf queryWeight;
    Eigen::MatrixXf keyWeight;
    Eigen::MatrixXf valueWeight;
    Eigen::MatrixXf outputWeight;
    Eigen::MatrixXf ffn1Weight;
    Eigen::MatrixXf ffn2Weight;

    // Hyperparameters
    size_t embeddingDim;
    size_t contextLen;
    size_t vocabularySize;
    int numLayers;
    int numHeads;

    // Random number generator
    std::mt19937 rng;

    // ================== Model Components ==================
    Eigen::MatrixXf calculateAttention(const Eigen::MatrixXf &Q, const Eigen::MatrixXf &K, const Eigen::MatrixXf &V) const
    {
        Eigen::MatrixXf scores = Q * K.transpose() / std::sqrt(static_cast<float>(Q.cols()));
        Eigen::MatrixXf attention = scores.array().exp();
        attention = attention.array().colwise() / attention.rowwise().sum().array();
        return attention * V;
    }

    Eigen::MatrixXf layerNorm(const Eigen::MatrixXf &input) const
    {
        Eigen::VectorXf mean = input.rowwise().mean();
        Eigen::VectorXf variance = (input.colwise() - mean).array().square().rowwise().mean();
        return (input.colwise() - mean).array().colwise() / (variance.array() + 1e-5).sqrt().array();
    }

    Eigen::MatrixXf feedForward(const Eigen::MatrixXf &input) const
    {
        Eigen::MatrixXf hidden = (input * ffn1Weight).array().max(0.0f); // ReLU
        return hidden * ffn2Weight;
    }

    Eigen::MatrixXf multiHeadAttention(const Eigen::MatrixXf &inputEmb) const
    {
        const size_t headDim = embeddingDim / numHeads;
        Eigen::MatrixXf result(inputEmb.rows(), embeddingDim);

        for (int h = 0; h < numHeads; ++h)
        {
            Eigen::MatrixXf Q = inputEmb * queryWeight.block(0, h * headDim, embeddingDim, headDim);
            Eigen::MatrixXf K = inputEmb * keyWeight.block(0, h * headDim, embeddingDim, headDim);
            Eigen::MatrixXf V = inputEmb * valueWeight.block(0, h * headDim, embeddingDim, headDim);

            Eigen::MatrixXf headOutput = calculateAttention(Q, K, V);

            result.block(0, h * headDim, inputEmb.rows(), headDim) = headOutput;
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
            word.erase(std::remove_if(word.begin(), word.end(),
                                      [](char c) { return std::ispunct(c); }),
                       word.end());
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);

            if (word.empty())
                continue;

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

    Eigen::MatrixXf getInputEmbeddings(const std::vector<size_t> &tokens) const
    {
        Eigen::MatrixXf inputEmb(tokens.size(), embeddingDim);

        for (size_t i = 0; i < tokens.size(); ++i)
        {
            if (tokens[i] >= vocabularySize)
                continue;
            inputEmb.row(i) = embeddings.row(tokens[i]);
        }

        for (size_t i = 0; i < tokens.size(); ++i)
        {
            for (size_t j = 0; j < embeddingDim; ++j)
            {
                float angle = i / std::pow(10000.0f, (2.0f * (j / 2)) / embeddingDim);
                if (j % 2 == 0)
                {
                    inputEmb(i, j) += std::sin(angle);
                }
                else
                {
                    inputEmb(i, j) += std::cos(angle);
                }
            }
        }
        return inputEmb;
    }

public:
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

        std::vector<std::pair<std::string, int>> wordFreq(wordCounts.begin(), wordCounts.end());
        std::sort(wordFreq.begin(), wordFreq.end(),
                  [](auto &a, auto &b)
                  { return a.second > b.second; });

        const size_t maxVocab = 5000;
        vocabularySize = std::min(wordFreq.size(), maxVocab) + 2; // +2 for special tokens
        idx2word.resize(vocabularySize);
        for (size_t i = 0; i < vocabularySize - 2; ++i)
        {
            word2idx[wordFreq[i].first] = i + 2;
            idx2word[i + 2] = wordFreq[i].first;
        }

        std::normal_distribution<float> dist(0.0f, 0.02f);
        auto init = [&]
        { return dist(rng); };

        embeddings = Eigen::MatrixXf::NullaryExpr(vocabularySize, embeddingDim, init);
        queryWeight = Eigen::MatrixXf::NullaryExpr(embeddingDim, embeddingDim, init);
        keyWeight = Eigen::MatrixXf::NullaryExpr(embeddingDim, embeddingDim, init);
        valueWeight = Eigen::MatrixXf::NullaryExpr(embeddingDim, embeddingDim, init);
        outputWeight = Eigen::MatrixXf::NullaryExpr(embeddingDim, vocabularySize, init);
        ffn1Weight = Eigen::MatrixXf::NullaryExpr(embeddingDim, 4 * embeddingDim, init);
        ffn2Weight = Eigen::MatrixXf::NullaryExpr(4 * embeddingDim, embeddingDim, init);
    }

    Eigen::MatrixXf forward(const std::vector<size_t> &tokens) const
    {
        if (tokens.empty())
            return Eigen::MatrixXf(0, 0);

        Eigen::MatrixXf x = getInputEmbeddings(tokens);

        for (int layer = 0; layer < numLayers; ++layer)
        {
            Eigen::MatrixXf attn = multiHeadAttention(x);
            x = x + attn;
            x = layerNorm(x);

            Eigen::MatrixXf ff = feedForward(x);
            x = x + ff;
            x = layerNorm(x);
        }

        return x * outputWeight;
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
            if (tokens.size() > contextLen)
            {
                tokens.erase(tokens.begin(), tokens.end() - contextLen);
            }

            Eigen::MatrixXf logits = forward(tokens);
            size_t lastPos = tokens.size() - 1;

            std::vector<float> probs(vocabularySize);
            for (size_t j = 0; j < vocabularySize; ++j)
            {
                probs[j] = std::exp(logits(lastPos, j) / temp);
            }

            float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
            for (auto &p : probs)
                p /= sum;

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

    std::vector<std::pair<std::string, float>> getRelatedWords(const std::string &word, int topN = 10) const
    {
        std::vector<std::pair<std::string, float>> results;
        if (!word2idx.count(word))
            return results;
    
        const size_t wordIdx = word2idx.at(word);
        Eigen::VectorXf wordVec = embeddings.row(wordIdx);
    
        std::vector<std::pair<float, std::string>> similarities;
        for (size_t i = 0; i < vocabularySize; ++i)
        {
            if (i == wordIdx)
                continue;
    
            Eigen::VectorXf currVec = embeddings.row(i);
            float cosine = wordVec.dot(currVec) / (wordVec.norm() * currVec.norm());
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