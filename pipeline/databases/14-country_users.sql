-- Write a SQL script that creates a table users following these requirements

CREATE TABLE IF NOT EXISTS users (
	id INT NOT NULL AUTO_INCREMENT UNIQUE,
	email VARCHAR(255) NOT NULL UNIQUE,
	name VARCHAR(255),
	country ENUM('US', 'CO', 'TN') NOT NULL DEFAULT 'US',
	PRIMARY KEY (id)
);
