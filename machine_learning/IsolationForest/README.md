# Isolation Forest
### Video Link: https://youtu.be/s0Y5PaVnAWU?si=kwtcObbrsWvQLIXt

### Description
🌲 What Is Isolation Forest?

Isolation Forest is an unsupervised anomaly detection algorithm. Unlike most algorithms that learn what “normal” is and then measure how far away something is, Isolation Forest works differently:

    Anomalies are easier to isolate.

It builds many random decision trees and observes how quickly a data point can be isolated.
🔍 Key Intuition

    Normal points are part of a "dense cluster" — you need many splits to isolate them.

    Anomalies (outliers) are few and far between — they get isolated with fewer splits.

So:

    🟢 Benign command lines → take many tree splits to isolate (they're common)

    🔴 Malicious/weird ones → get separated quickly (they’re different)

🧱 How It Works (Step by Step)

Let’s say you have a TF-IDF matrix of command-line logs.
1. Build multiple trees

    Each tree is built using a random subset of the data.

    At each node:

        Pick a random feature (e.g. "powershell", "cmd", etc.).

        Choose a random split value.

        Split the data.

Do this recursively until:

    The data point is isolated, or

    A maximum depth is reached.

2. Measure path length

    For each data point, compute the average path length across all trees:

        How many splits were needed to isolate it?

3. Score anomaly

    Shorter average path length = more anomalous.

    The score is normalized between ~0 and 1:

        Near 1 → anomalous

        Near 0 → normal

📊 Example

Imagine this 2D dataset:

. . . . . . . x . . .
. . . . . . . . . . .
. . . . . . . . . . .

    The dots (.) are normal command lines (dense cluster)

    The x is a weird one — maybe a powershell -enc ABCDEFG line

Isolation Trees will isolate the x much faster than any dot.
🔐 Why It Works for Logs

For your case:

    Normal command lines might include:

"C:\Program Files\App\app.exe"
"cmd /c dir"
"explorer.exe C:\Users"

Malicious ones might look like:

    "powershell -enc <long blob>"
    "cmd /c whoami & netstat"

These weird patterns will have rare tokens → they’re easier to isolate in the TF-IDF feature space.
⚙️ Summary
Concept	Explanation
Key Idea	Anomalies are easier to isolate via random splits
Core Mechanism	Builds random trees and measures how fast a point is separated
Anomaly Signal	Shorter average path length across trees
Why it works	Outliers differ in enough dimensions to get quickly partitioned