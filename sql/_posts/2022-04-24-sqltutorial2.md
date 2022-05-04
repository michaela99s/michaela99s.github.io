---
layout: post
title: tutorial2
description: >
  Howdy! This is an example blog post that shows several types of HTML content supported in this theme.
sitemap: false
hide_last_modified: true
---

```sql
mysql> CREATE TABLE shop (
    -> article INT UNSIGNED  DEFAULT '0000' NOT NULL,
    -> dealer  CHAR(20)      DEFAULT ''     NOT NULL,
    -> price   DECIMAL(16,2) DEFAULT '0.00' NOT NULL,
    -> PRIMARY KEY (article, dealer)
    -> );
Query OK, 0 rows affected (0.05 sec)

mysql> INSERT INTO shop VALUES
    -> (1, 'A', 3.45),
    -> (1, 'B', 3.99),
    -> (2, 'A', 10.99),
    -> (3, 'B', 1.45),
    -> (3, 'C', 1.69),
    -> (3, 'D', 1.25),
    -> (4, 'D', 19.95);
Query OK, 7 rows affected (0.02 sec)
Records: 7  Duplicates: 0  Warnings: 0

mysql> SELECT * FROM shop ORDER BY article;
+---------+--------+-------+
| article | dealer | price |
+---------+--------+-------+
|       1 | A      |  3.45 |
|       1 | B      |  3.99 |
|       2 | A      | 10.99 |
|       3 | B      |  1.45 |
|       3 | C      |  1.69 |
|       3 | D      |  1.25 |
|       4 | D      | 19.95 |
+---------+--------+-------+
7 rows in set (0.00 sec)s: 0  Warnings: 0
```

```sql
mysql> SELECT MAX(article) AS article FROM shop;
+---------+
| article |
+---------+
|       4 |
+---------+
1 row in set (0.01 sec)
```

```sql
mysql> SELECT article, dealer, price FROM shop
    -> WHERE price = (SELECT MAX(price) FROM shop);
+---------+--------+-------+
| article | dealer | price |
+---------+--------+-------+
|       4 | D      | 19.95 |
+---------+--------+-------+
1 row in set (0.01 sec)

mysql> SELECT s1.article, s1.dealer, s1.price FROM shop s1
    -> LEFT JOIN shop s2 ON s1.price < s2.price
    -> WHERE s2.article IS NULL;
+---------+--------+-------+
| article | dealer | price |
+---------+--------+-------+
|       4 | D      | 19.95 |
+---------+--------+-------+
1 row in set (0.00 sec)

mysql> SELECT article, dealer, price FROM shop
    -> ORDER BY price DESC
    -> LIMIT 1;
+---------+--------+-------+
| article | dealer | price |
+---------+--------+-------+
|       4 | D      | 19.95 |
+---------+--------+-------+
1 row in set (0.01 sec)
```

```sql
mysql> SELECT article, MAX(price) AS price FROM shop
    -> GROUP BY article
    -> ORDER BY article;
+---------+-------+
| article | price |
+---------+-------+
|       1 |  3.99 |
|       2 | 10.99 |
|       3 |  1.69 |
|       4 | 19.95 |
+---------+-------+
4 rows in set (0.01 sec)
```

```sql
mysql> SELECT article, dealer, price FROM shop s1
    -> WHERE price = (SELECT MAX(s2.price) FROM shop s2
    -> WHERE s1.article = s2.article)
    -> ORDER BY article;
+---------+--------+-------+
| article | dealer | price |
+---------+--------+-------+
|       1 | B      |  3.99 |
|       2 | A      | 10.99 |
|       3 | C      |  1.69 |
|       4 | D      | 19.95 |
+---------+--------+-------+
4 rows in set (0.00 sec)

mysql> SELECT s1.article, dealer, s1.price FROM shop s1
    -> JOIN (
    ->  SELECT article, MAX(price) AS price FROM shop
    ->  GROUP BY article) AS s2
    -> ON s1.article = s2.article AND s1.price = s2.price
    -> ORDER BY article;
+---------+--------+-------+
| article | dealer | price |
+---------+--------+-------+
|       1 | B      |  3.99 |
|       2 | A      | 10.99 |
|       3 | C      |  1.69 |
|       4 | D      | 19.95 |
+---------+--------+-------+
4 rows in set (0.01 sec)

mysql> WITH s1 AS (
    ->  SELECT article, dealer, price,
    ->         RANK() OVER (PARTITION BY article
    ->                          ORDER BY price DESC
    ->                     ) AS `Rank`
    ->         FROM shop
    -> )
    -> SELECT article, dealer, price FROM s1
    -> WHERE `Rank` = 1
    -> ORDER BY article;
+---------+--------+-------+
| article | dealer | price |
+---------+--------+-------+
|       1 | B      |  3.99 |
|       2 | A      | 10.99 |
|       3 | C      |  1.69 |
|       4 | D      | 19.95 |
+---------+--------+-------+
4 rows in set (0.00 sec)

```

```sql
mysql> SELECT @min_price:=MIN(price), @max_price:=MAX(price) FROM shop;
+------------------------+------------------------+
| @min_price:=MIN(price) | @max_price:=MAX(price) |
+------------------------+------------------------+
|                   1.25 |                  19.95 |
+------------------------+------------------------+
1 row in set, 2 warnings (0.00 sec)

mysql> SELECT * FROM shop WHERE price=@min_price OR price=@max_price;
+---------+--------+-------+
| article | dealer | price |
+---------+--------+-------+
|       3 | D      |  1.25 |
|       4 | D      | 19.95 |
+---------+--------+-------+
2 rows in set (0.00 sec)
```

```sql
mysql> CREATE TABLE parent (
    -> id INT NOT NULL,
    -> PRIMARY KEY (id)
    -> ) ENGINE = INNODB;
Query OK, 0 rows affected (0.04 sec)

mysql> CREATE TABLE child (
    -> id INT,
    -> parent_id INT,
    -> INDEX par_ind (parent_id),
    -> FOREIGN KEY (parent_id)
    -> REFERENCES parent (id)
    -> ) ENGINE = INNODB;
Query OK, 0 rows affected (0.06 sec)
```

```sql
mysql> INSERT INTO parent (id) VALUES (1);
Query OK, 1 row affected (0.01 sec)

mysql> SELECT * FROM parent;
+----+
| id |
+----+
|  1 |
+----+
1 row in set (0.00 sec)
```

```sql
mysql> INSERT INTO child (id, parent_id) VALUES (1,1);
Query OK, 1 row affected (0.01 sec)

mysql> INSERT INTO child (id, parent_id) VALUES (2,2);
ERROR 1452 (23000): Cannot add or update a child row: a foreign key constraint fails (`practice`.`child`, CONSTRAINT `child_ibfk_1` FOREIGN KEY (`parent_id`) REFERENCES `parent` (`id`))
mysql> DELETE FROM parent WHERE id VALUES = 1;
ERROR 1064 (42000): You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use near 'VALUES = 1' at line 1

```

```sql
mysql> CREATE TABLE child (
    -> id INT,
    -> parent_id INT,
    -> INDEX par_id (parent_id),
    -> FOREIGN KEY (parent_id)
    -> REFERENCES parent(id)
    -> ON UPDATE CASCADE
    -> ON DELETE CASCADE
    -> ) ENGINE = INNODB;
Query OK, 0 rows affected (0.05 sec)
```

```sql
mysql> INSERT INTO child (id, parent_id) VALUES(1,1), (2,1), (3,1);
Query OK, 3 rows affected (0.01 sec)
Records: 3  Duplicates: 0  Warnings: 0

mysql> SELECT * FROM child;
+------+-----------+
| id   | parent_id |
+------+-----------+
|    1 |         1 |
|    2 |         1 |
|    3 |         1 |
+------+-----------+
3 rows in set (0.00 sec)
```

```sql
mysql> UPDATE parent SET id = 2 WHERE id = 1;
Query OK, 1 row affected (0.01 sec)
Rows matched: 1  Changed: 1  Warnings: 0

mysql> SELECT * FROM parent;
+----+
| id |
+----+
|  2 |
+----+
1 row in set (0.01 sec)

mysql> SELECT * FROM child;
+------+-----------+
| id   | parent_id |
+------+-----------+
|    1 |         2 |
|    2 |         2 |
|    3 |         2 |
+------+-----------+
3 rows in set (0.01 sec)

```

```sql
mysql> DELETE FROM parent WHERE id = 2;
Query OK, 1 row affected (0.01 sec)

mysql> SELECT * FROM child;
Empty set (0.00 sec)
```

```sql
mysql> CREATE TABLE t1(year YEAR, month INT UNSIGNED,
    ->                 day INT UNSIGNED);
Query OK, 0 rows affected (0.05 sec)

mysql> INSERT INTO t1 VALUES (2000,1,1), (2000,1,20), (2000,1,30),
    ->                       (2000,2,2), (2000,2,23), (2000,2,23);
Query OK, 6 rows affected (0.02 sec)
Records: 6  Duplicates: 0  Warnings: 0

mysql> SELECT year, month, BIT_COUNT(BIT_OR(1<<day)) AS days FROM t1
    -> GROUP BY year, month;
+------+-------+------+
| year | month | days |
+------+-------+------+
| 2000 |     1 |    3 |
| 2000 |     2 |    2 |
+------+-------+------+
2 rows in set (0.01 sec)
```

```sql
mysql> CREATE TABLE animals (
    -> id MEDIUMINT NOT NULL AUTO_INCREMENT,
    -> name CHAR(30) NOT NULL,
    -> PRIMARY KEY (id)
    -> );
Query OK, 0 rows affected (0.04 sec)

mysql> INSERT INTO animals (name) VALUES
    -> ('dog'), ('cat'), ('penguin'),
    -> ('lax'), ('whale'), ('ostrich');
Query OK, 6 rows affected (0.02 sec)
Records: 6  Duplicates: 0  Warnings: 0

mysql> SELECT * FROM animals;
+----+---------+
| id | name    |
+----+---------+
|  1 | dog     |
|  2 | cat     |
|  3 | penguin |
|  4 | lax     |
|  5 | whale   |
|  6 | ostrich |
+----+---------+
6 rows in set (0.00 sec)
```

```sql
mysql> INSERT INTO animals (id, name) VALUES (0, 'groundhog');
Query OK, 1 row affected (0.01 sec)

mysql> INSERT INTO animals (id, name) VALUES (NULL, 'squirrel');
Query OK, 1 row affected (0.01 sec)

mysql> INSERT INTO animals (id, name) VALUES (100, 'rabbit');
Query OK, 1 row affected (0.01 sec)

mysql> INSERT INTO animals (id, name) VALUES (NULL, 'mouse');
Query OK, 1 row affected (0.01 sec)

```

```sql
mysql> SELECT * FROM animals;
+-----+-----------+
| id  | name      |
+-----+-----------+
|   1 | dog       |
|   2 | cat       |
|   3 | penguin   |
|   4 | lax       |
|   5 | whale     |
|   6 | ostrich   |
|   7 | groundhog |
|   8 | squirrel  |
| 100 | rabbit    |
| 101 | mouse     |
+-----+-----------+
10 rows in set (0.00 sec)

```

```sql
mysql> DROP TABLE animals;
Query OK, 0 rows affected (0.02 sec)

mysql> CREATE TABLE animals (
    -> grp ENUM('fish', 'mammal', 'bird') NOT NULL,
    -> id MEDIUMINT NOT NULL AUTO_INCREMENT,
    -> name CHAR(30) NOT NULL,
    -> PRIMARY KEY (grp, id)
    -> ) ENGINE = MyISAM;
Query OK, 0 rows affected (0.02 sec)

mysql> INSERT INTO animals (grp, name) VALUES
    -> ('mammal', 'dog'), ('mammal', 'cat'),
    -> ('bird', 'penguin'), ('fish', 'lax'), ('mammal', 'whale'),
    -> ('bird', 'ostrich');
Query OK, 6 rows affected (0.01 sec)
Records: 6  Duplicates: 0  Warnings: 0

mysql> SELECT * FROM animals ORDER by grp, id;
+--------+----+---------+
| grp    | id | name    |
+--------+----+---------+
| fish   |  1 | lax     |
| mammal |  1 | dog     |
| mammal |  2 | cat     |
| mammal |  3 | whale   |
| bird   |  1 | penguin |
| bird   |  2 | ostrich |
+--------+----+---------+
6 rows in set (0.00 sec)
```
