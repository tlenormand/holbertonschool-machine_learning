# Docker

Build :

<code>docker compose up --build</code>

Connect :

<code>docker compose run math bash</code>


# Usage

## Run Python File

<code>python3 <filename.py></code>

## Run Python File in Docker

<code>docker compose run math python3 <filename.py></code>

## Run Python File in Docker with Input

<code>echo -e "<input>" | docker compose run math python3 <filename.py></code>

## Run Python File in Docker with Input and Output

<code>echo -e "<input>" | docker compose run math python3 <filename.py> > <output.txt></code>


# Projects

## [0x00-linear_algebra](./linear_algebra)