
# Vector Database

This repository contains an implementation of a Vector Database, a specialized database designed for efficient storage and retrieval of high-dimensional vectors. Ideal for applications such as machine learning, similarity search, and data analysis.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JledBgKF_maHA1qxY_FCgVx0O8Qgxvys)

## Features

- **Efficient Vector Storage:** Store and manage high-dimensional vectors in a space-efficient manner.
- **Fast Retrieval:** Implement efficient algorithms for retrieving vectors based on similarity search.
- **Scalability:** Design the database to scale seamlessly as the volume of vectors increases.
- **Customizable:** Easily adapt the database implementation to specific use cases and requirements.

## Getting Started

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/Vector-Database.git
    cd Vector-Database
    ```

2. Explore the implementation and review the documentation for usage instructions.

## Usage

```python
python main.py
```

## Structure
```
Vector-Database-Project/
├── notebooks/
│   └── vector_database.ipynb
├── src/
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── without_gpu.py
│   │   └── with_gpu.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── annoy_database.py
│   │   └── faiss_database.py
│   ├── query/
│   │   ├── __init__.py
│   │   └── query.py
│   └── utils/
│       ├── __init__.py
│       └── helper_functions.py
├── data/
│   └── combined_questions_filtered.json
├── tests/
│   └── test_queries.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Contributions

Contributions are welcome! Feel free to open issues, submit pull requests, or suggest improvements.

## License

This project is licensed under the MIT License.

