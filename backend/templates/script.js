document.getElementById("rec-button").addEventListener("click", function() {
  // Get input values and process them
      const titleRaw = document.getElementById("bookTitle").value;
      const authorRaw = document.getElementById("author").value;
      const categoryRaw = document.getElementById("category").value;

      const titles = titleRaw.split(",").map(item => item.trim()).filter(item => item !== "");
      const authors = authorRaw.split(",").map(item => item.trim()).filter(item => item !== "");
      const categories = categoryRaw.split(",").map(item => item.trim()).filter(item => item !== "");

      // Display loading message
      const container = document.getElementById('books-container');
      container.innerHTML = '<p class="loading">Loading...</p>';

      setTimeout(() => {
          fetch('/getbooks', {
              method: "POST",
              headers: {
                  "Content-Type": "application/json",
              },
              body: JSON.stringify({
                  titles: titles,
                  authors: authors,
                  categories: categories
              }),
          })
          .then(response => response.json())
          .then(data => {
              // Clear the loading message
              container.innerHTML = '';

              data.forEach(book => {
                  // Create book card container
                  const card = document.createElement('div');
                  card.className = 'book-card';

                  const left = document.createElement('div');
                  left.className = 'card-left';

                  // Image
                  const image = document.createElement('img');
                  image.className = 'book-image';
                  image.src = "{{ url_for('static', filename='images/open-book.jpeg') }}";
                  image.alt = book.title;
                  left.appendChild(image);

                  const similar = document.createElement('p');
                  similar.className = 'sim';
                  similar.textContent = "Similarity Score";
                  left.appendChild(similar);

                  // Similarity Score
                  const score = document.createElement('p');
                  score.className = 'sim-score';
                  score.textContent = book.score.toFixed(2);
                  left.appendChild(score);

                  card.appendChild(left);

                  // Container for text information
                  const info = document.createElement('div');
                  info.className = 'book-info';

                  // Title
                  const title = document.createElement('div');
                  title.className = 'book-title';
                  title.textContent = book.title ?? "Title Not Available";
                  info.appendChild(title);

                  // Categories 
                  const categoriesEl = document.createElement('p');
                  categoriesEl.className = 'book-meta';
                  categoriesEl.textContent = 'Categories: ' + ((Array.isArray(book.categories) ? book.categories.join(', ') : book.categories) ?? 'Uncategorized');
                  info.appendChild(categoriesEl);

                  // Author(s)
                  const authorsEl = document.createElement('p');
                  authorsEl.className = 'book-meta';
                  authorsEl.textContent = 'Author(s): ' + ((Array.isArray(book.authors) ? book.authors.join(', ') : book.authors) ?? 'Author Not Available');
                  info.appendChild(authorsEl);

                  // Published Date
                  const publishedDate = document.createElement('p');
                  publishedDate.className = 'book-meta';
                  publishedDate.textContent = 'Published Date: ' + (book.publishDate ?? 'Unknown');
                  info.appendChild(publishedDate);

                  // Description
                  const description = document.createElement('p');
                  description.className = 'book-description';
                  description.textContent = book.description.toString() ?? 'Description Unavailable';
                  info.appendChild(description);

                  card.appendChild(info);
                  container.appendChild(card);
              });
          })
          .catch(error => {
              console.error('Error fetching books:', error);
              container.innerHTML = '<p class="error">Error loading books. Please try again.</p>';
          });
      }, 1000); 
  });