#include <vector>
#include <tuple>
#include <map>
#include <algorithm>
#include <array>
#include <random>
#include <ctime>
#include <thread>
#include <chrono>
#include <mutex>

class BrainMatrix {
public:
    std::vector<std::vector<std::vector<Neuron*>>> grid;

    BrainMatrix(int sizeX, int sizeY, int sizeZ) {
        grid.resize(sizeX, std::vector<std::vector<Neuron*>>(sizeY, std::vector<Neuron*>(sizeZ, nullptr)));
    }

    void placeNeuron(Neuron* neuron, int x, int y, int z) {
        if (grid[x][y][z] == nullptr) {
            grid[x][y][z] = neuron;
        }
    }

    void removeNeuron(int x, int y, int z) {
        if (grid[x][y][z] != nullptr) {
            delete grid[x][y][z];
            grid[x][y][z] = nullptr;
        }
    }

    bool isLocationFree(int x, int y, int z) {
        return grid[x][y][z] == nullptr;
    }

    Neuron* getNeuron(int x, int y, int z) {
        return grid[x][y][z];
    }
};

class Neuron {
public:
    int x, y, z;
    Neuron(int x, int y, int z) : x(x), y(y), z(z) {};
    virtual void activate(int input) {}
    virtual void resetCanFire() {}
};

class GenericNeuron : public Neuron {
private:
    bool canFire = true;
    std::mutex fireMutex;  // Mutex to synchronize access to canFire
    bool neighborsFound = false;
    bool hasChild = false;
    int maxLevel = 10000;
    int importance = 30;
    int age = 1;
    int ageCount = 0;
    int reverseAgeCount = 0;
    int blacklistResetCounter = 0;
    std::vector<std::tuple<Neuron*, int, int, int>> recipientCandidates;
    std::vector<std::tuple<Neuron*, int, int, int, int>> recipients;
    std::map<std::tuple<Neuron*, int, int, int>, int> recipientStrikes;
    std::vector<std::tuple<Neuron*, int, int, int>> blacklist;
    BrainMatrix* grid;
    Neuron* parent;

public:
    static std::vector<GenericNeuron*> instances;

    GenericNeuron(int x, int y, int z, BrainMatrix* grid, Neuron* parent = nullptr)
        : Neuron(x, y, z), grid(grid), parent(parent) {
        instances.push_back(this);
        grid->placeNeuron(this, x, y, z);
    }

    ~GenericNeuron() {
        auto it = std::find(instances.begin(), instances.end(), this);
        if (it != instances.end()) {
            instances.erase(it);
        }
        grid->removeNeuron(x, y, z);
    }

    void commitSudoku() {
        delete this;
    }

    void updateRecipientAges(bool increment, int amount) {
        for (auto& recipient : recipients) {
            int& conAge = std::get<4>(recipient);  // Access the connection age
            if (increment) {
                if (conAge < maxLevel) {
                    conAge += amount;
                }
            }
            else {
                conAge -= amount;
            }
        }
    }

    void train(bool punish, int amount) {
        if (punish) {
            importance -= amount;
            if (importance <= 0) {
                commitSudoku();
                return;
            }
            reverseAgeCount++;
            if (reverseAgeCount > 5) {
                age--;
                reverseAgeCount = 0;
            }
            updateRecipientAges(false, amount);
        }
        else {
            if (importance < maxLevel) {
                importance += amount;
            }
            ageCount++;
            if (ageCount >= 5) {
                if (age < maxLevel) {
                    age++;
                }
                ageCount = 0;
            }
            updateRecipientAges(true, amount);
        }
    }

    static void trainAll(bool punish, int amount) {
        for (auto instance : instances) {
            instance->train(punish, amount);
        }
    }

    //get list of nearby neuron positions other than parent
    void getCandidates() {
        recipientCandidates.clear();  // Clear previous candidates

        std::array<int, 27> dx = { 1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, -1, 0 };
        std::array<int, 27> dy = { 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1 };
        std::array<int, 27> dz = { 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, -1, 1, 0 };

        int parentX = (parent != nullptr) ? parent->x : -1;
        int parentY = (parent != nullptr) ? parent->y : -1;
        int parentZ = (parent != nullptr) ? parent->z : -1;

        for (int i = 0; i < 27; ++i) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            int nz = z + dz[i];

            if (!(nx == parentX && ny == parentY && nz == parentZ)) {  // Exclude parent position
                Neuron* nearbyNeuron = grid->getNeuron(nx, ny, nz);
                if (nearbyNeuron != nullptr) {
                    addRecipient(nearbyNeuron, nx, ny, nz);
                }
            }
        }
    }

    //add a neuron to the list of signal recipients
    void addRecipient(Neuron* neuron, int x, int y, int z) {
        recipients.emplace_back(neuron, x, y, z, 1);
    }

    //removing missing neurons or bad connections from list of signal recipients
    void updateRecipients() {
        recipients.erase(std::remove_if(recipients.begin(), recipients.end(),
            [this](const std::tuple<Neuron*, int, int, int, int>& recipient) {
                Neuron* neuron = std::get<0>(recipient);
                int x = std::get<1>(recipient);
                int y = std::get<2>(recipient);
                int z = std::get<3>(recipient);
                int conAge = std::get<4>(recipient);
                if (grid->getNeuron(x, y, z) == neuron && conAge <= 0) {
                    auto key = std::make_tuple(neuron, x, y, z);
                    if (recipientStrikes.find(key) == recipientStrikes.end()) {
                        recipientStrikes[key] = 1;
                    }
                    else {
                        recipientStrikes[key]++;
                        if (recipientStrikes[key] >= 3) {
                            blacklist.emplace_back(neuron, x, y, z);
                            recipientStrikes.erase(key);
                        }
                    }
                }
                return grid->getNeuron(x, y, z) != neuron || conAge <= 0;  // Remove if neuron is not valid or age <= 0
            }), recipients.end());
    }

    bool connectNearbyNeuron() {
        // Create a vector to hold occupied positions that are not in the recipient list
        std::vector<std::tuple<Neuron*, int, int, int>> eligiblePositions;

        // Populate eligiblePositions with candidates that are not in the recipient list
        for (const auto& candidate : recipientCandidates) {
            Neuron* neuron = std::get<0>(candidate);
            int nx = std::get<1>(candidate);
            int ny = std::get<2>(candidate);
            int nz = std::get<3>(candidate);

            // Check if the candidate is already a recipient
            bool isRecipient = std::any_of(recipients.begin(), recipients.end(),
                [neuron, nx, ny, nz](const std::tuple<Neuron*, int, int, int, int>& recipient) {
                    return std::get<0>(recipient) == neuron &&
                        std::get<1>(recipient) == nx &&
                        std::get<2>(recipient) == ny &&
                        std::get<3>(recipient) == nz;
                });

            if (!isRecipient && grid->getNeuron(nx, ny, nz) != nullptr) {
                eligiblePositions.emplace_back(neuron, nx, ny, nz);
            }
        }

        // If there are eligible positions, randomly select one and add it to the recipient list
        if (!eligiblePositions.empty()) {
            std::srand(std::time(nullptr));
            int randomIndex = std::rand() % eligiblePositions.size();
            auto [neuron, nx, ny, nz] = eligiblePositions[randomIndex];

            // Add the selected neuron to the recipient list with a starting connection age
            addRecipient(neuron, nx, ny, nz);
            return true;
        }

        return false;  // No eligible positions available
    }

    bool placeNearbyNeuron() {
        // Create a vector to hold empty positions
        std::vector<std::tuple<int, int, int>> emptyPositions;

        // Populate emptyPositions with positions from recipientCandidates that are empty
        for (const auto& candidate : recipientCandidates) {
            int nx = std::get<1>(candidate);
            int ny = std::get<2>(candidate);
            int nz = std::get<3>(candidate);

            if (grid->isLocationFree(nx, ny, nz)) {
                emptyPositions.emplace_back(nx, ny, nz);
            }
        }

        // If there are empty positions, randomly select one to place a new neuron
        if (!emptyPositions.empty()) {
            std::srand(std::time(nullptr));
            int randomIndex = std::rand() % emptyPositions.size();
            auto [nx, ny, nz] = emptyPositions[randomIndex];

            // Create and place a new neuron in the selected empty position
            Neuron* newNeuron = new GenericNeuron(nx, ny, nz, grid, this);
            addRecipient(newNeuron, nx, ny, nz);  // Add new recipient connection
            return true;
        }

        return false;  // No empty positions available
    }

    // Function to simulate a coin flip
    bool coinFlip() {
        std::srand(std::time(nullptr));  // Seed the random number generator
        return std::rand() % 2 == 0;  // Returns true for heads, false for tails
    }

    void resetCanFire() override {
        std::lock_guard<std::mutex> lock(fireMutex);  // Lock mutex during update
        canFire = true;
    }

    //TODO:
    //need a way to allow for multiple calls to activate to accumulate a coimbined input value
    //to allow for semi-asynchonous activation
    //maybe

    //maybe add a pre-activate function which spins up a seperate thread to count down and if the counter ends before
    //the function is called again, send the value
    //otherwise, update the value and reset the counter
    //if it's possible to make calls to a seperate task thread

    void activate(int input) override {  // Override with one parameter
        if (importance <= 0) {
            commitSudoku();
        }

        if (!neighborsFound) {
            getCandidates();
            neighborsFound = true;
        }

        if (input > 55) {
            if (canFire) {
                importance++;
                updateRecipients();
                blacklistResetCounter++;
                if (blacklistResetCounter > 10000) {
                    blacklist.clear();
                    blacklistResetCounter = 0;
                }

                if (recipients.size() < (recipientCandidates.size() - blacklist.size())) {

                    if (!hasChild) {
                        bool tryCreateChild = placeNearbyNeuron();
                        if (tryCreateChild) {
                            hasChild = true;
                        }
                    }
                    else {
                        bool heads = coinFlip();
                        if (heads) {
                            bool tryCreateChild = placeNearbyNeuron();
                        }
                        else {
                            bool tryConnectNeuron = connectNearbyNeuron();
                        }
                    }

                    if (recipients.empty()) {
                        bool foundOpenPosition = placeNearbyNeuron();
                        bool connectOtherNeuron = connectNearbyNeuron();
                        if (!foundOpenPosition && !connectOtherNeuron) {
                            commitSudoku();
                        }
                    }

                }
                int output = input + age;
                int scalingFactorPercent = 10; //percent by which to reduce final value per each output
                int scaleAmountPercent = recipients.size() * scalingFactorPercent;
                int multiplyByPercent = 100 - scaleAmountPercent;
                int finalOutput = (output * multiplyByPercent) / 100;
                for (auto& recipient : recipients) {
                    std::get<0>(recipient)->activate(finalOutput);  // Call the activate method on recipients
                }
                //disable firing for a period
                std::lock_guard<std::mutex> lock(fireMutex);
                canFire = false;
                std::thread(restNeuron, this, 100).detach();
            }
        }
    }
};

// Initialize the static member
std::vector<GenericNeuron*> GenericNeuron::instances;

//todo: multiple primary neuron types. inputs and outputs.
//inputs connected to camera feed pixels or similar. maybe preprocessed
//outputs to a display or similar;
//inputs connected to segmented frequency inputs from an audio source.
//mirrored outputs to reversed version for a speaker output.
//basic structure expected to have these networks cross eachother to allow for interconection, in the "space" between inputs and outputs
//optimal size unknown
//optimal starting structure unknown
//planned self reinforcement primary output neurons, one for reinforcement and one for punishing
//system should learn to reinforce it's own behavior in line with user actions
//self reinforcement intentionally less than manual training input
//added benefit when self reinforcement matches training reinforcement
//all numerical
//can explain in more detail
//not sure it would work though

class PrimaryNeuron : public Neuron {
public:
    PrimaryNeuron(int x, int y, int z) : Neuron(x, y, z) {}

    void activate(int dummy) {
        // Implementation for PrimaryNeuron activation
    }
};

// External function to reset canFire after a short rest
void restNeuron(GenericNeuron* neuron, int restTimeMs) {
    std::this_thread::sleep_for(std::chrono::milliseconds(restTimeMs));  // Wait for the specified time
    neuron->resetCanFire();  // Reset canFire after waiting
}


