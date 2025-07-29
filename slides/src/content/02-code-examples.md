## Code Examples

Let's look at some JavaScript:

```javascript
// Hello World function
function greetUser(name) {
    console.log(`Hello, ${name}!`);
    return `Welcome to the presentation, ${name}`;
}

// Usage example
const message = greetUser("Developer");
document.getElementById("output").innerHTML = message;
```

---

## CSS Styling Examplesadfjk

Here's some custom CSS:

```css
.highlight {
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
    color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

.code-container {
    background: #2d3748;
    border-left: 4px solid #4299e1;
    padding: 1rem;
}
```

---

## Python Example

Data processing with Python:

```python
import pandas as pd
import numpy as np

# Create sample data
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Calculate statistics
average_salary = df['salary'].mean()
print(f"Average salary: ${average_salary:,.2f}")
```