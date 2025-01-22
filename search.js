let abstracts = [];

fetch("abstracts.json")
  .then(response => response.json())
  .then(data => {
    abstracts = data;
  })
  .catch(error => console.error("Error loading abstracts:", error));

function performSearch(query) {
  const resultsDiv = document.getElementById("results");
  resultsDiv.innerHTML = "";

  if (query.trim() === "") {
    resultsDiv.innerHTML = "<p>Search abstracts with keywords.</p>";
    return;
  }

  const results = abstracts.filter(item =>
    item.abstract.toLowerCase().includes(query.toLowerCase())
  );

  if (results.length === 0) {
    resultsDiv.innerHTML = "<p>No results found.</p>";
  } else {
    results.forEach(result => {
      const resultItem = document.createElement("div");
      resultItem.className = "result";

      // for highlighting the keywords in every search result (abstract)
      const highlighted = result.abstract.replace(
        new RegExp(query, "gi"),
        match => `<span class="highlight">${match}</span>`
      );
      resultItem.innerHTML = highlighted;
      resultsDiv.appendChild(resultItem);
    });
  }
}

document.getElementById("searchInput").addEventListener("input", function () {
  performSearch(this.value);
});

const style = document.createElement("style");
style.textContent = `
  .highlight {
    background-color: yellow;
    font-weight: bold;
  }
`;
document.head.appendChild(style);