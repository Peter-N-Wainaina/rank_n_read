document.getElementById("rec-button").addEventListener("click", function() {
  alert("Feature coming soon! Personalized book recommendations will be available here.");

  fetch('/getbooks')
    .then(response => response.json())
    .then(data => {
      const container = document.getElementById('books-container');
      data.forEach(book => {
        // Create a container for each book.
        const card = document.createElement('div');
        card.className = 'book-card';
        
        // Create the title element.
        const title = document.createElement('div');
        title.className = 'book-title';
        title.textContent = book.title;
        card.appendChild(title);
        
        // Create the author element.
        const author = document.createElement('p');
        author.className = 'book-author';
        author.textContent = 'Author: ' + book.author;
        card.appendChild(author);
        
        // Create the year element.
        const year = document.createElement('p');
        year.className = 'book-year';
        year.textContent = 'Year: ' + book.year;
        card.appendChild(year);
        
        // Append the book card to the container.
        container.appendChild(card);
      });
    })
    .catch(error => console.error('Error fetching books:', error));
});