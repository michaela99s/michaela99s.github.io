---
layout: post
title: tutorial1
description: >
  Howdy! This is an example blog post that shows several types of HTML content supported in this theme.
sitemap: false
hide_last_modified: true
---

```sql
mysql USE menagerie
```

```sql
mysql SHOW TABLES;

Empty set (0.01 sec)
```

```sql
mysql CREATE TABLE pet (
    - name VARCHAR(20),
    - owner VARCHAR(20),
    - species VARCHAR(20),
    - sex CHAR(1),
    - birth DATE,
    - death DATE
    - );


Query OK, 0 rows affected (0.05 sec)

mysql SHOW TABLES;

+---------------------+
 Tables_in_menagerie 
+---------------------+
 pet                 
+---------------------+
1 row in set (0.01 sec)
```

```sql
mysql DESCRIBE pet;

+---------+-------------+------+-----+---------+-------+
 Field    Type         Null  Key  Default  Extra 
+---------+-------------+------+-----+---------+-------+
 name     varchar(20)  YES        NULL           
 owner    varchar(20)  YES        NULL           
 species  varchar(20)  YES        NULL           
 sex      char(1)      YES        NULL           
 birth    date         YES        NULL           
 death    date         YES        NULL           
+---------+-------------+------+-----+---------+-------+
6 rows in set (0.01 sec)
```

```sql
mysql LOAD DATA LOCAL INFILE "C:/notes/SQL/menagerie-dbpet.txt" INTO TABLE pet;
Query OK, 8 rows affected (0.02 sec)
Records 8  Deleted 0  Skipped 0  Warnings 0
```

```sql
mysql SELECT * FROM pet;

+----------+--------+---------+------+------------+------------+
 name      owner   species  sex   birth       death      
+----------+--------+---------+------+------------+------------+
 Fluffy    Harold  cat      f     1993-02-04  NULL       
 Claws     Gwen    cat      m     1994-03-17  NULL       
 Buffy     Harold  dog      f     1989-05-13  NULL       
 Fang      Benny   dog      m     1990-08-27  NULL       
 Bowser    Diane   dog      m     1979-08-31  1995-07-29 
 Chirpy    Gwen    bird     f     1998-09-11  NULL       
 Whistler  Gwen    bird     NULL  1997-12-09  NULL       
 Slim      Benny   snake    m     1996-04-29  NULL       
+----------+--------+---------+------+------------+------------+
8 rows in set (0.01 sec)
```

```sql
mysql INSERT INTO pet
    - VALUES ('Puffball', 'Diane', 'hamster', 'f', '1999-03-30', NULL);
Query OK, 1 row affected (0.01 sec)
```

```sql
mysql SELECT * FROM pet;

+----------+--------+---------+------+------------+------------+
 name      owner   species  sex   birth       death      
+----------+--------+---------+------+------------+------------+
 Fluffy    Harold  cat      f     1993-02-04  NULL       
 Claws     Gwen    cat      m     1994-03-17  NULL       
 Buffy     Harold  dog      f     1989-05-13  NULL       
 Fang      Benny   dog      m     1990-08-27  NULL       
 Bowser    Diane   dog      m     1979-08-31  1995-07-29 
 Chirpy    Gwen    bird     f     1998-09-11  NULL       
 Whistler  Gwen    bird     NULL  1997-12-09  NULL       
 Slim      Benny   snake    m     1996-04-29  NULL       
 Puffball  Diane   hamster  f     1999-03-30  NULL       
+----------+--------+---------+------+------------+------------+
9 rows in set (0.01 sec)
```

```sql
mysql UPDATE pet SET birth = '1989-08-31' WHERE name = 'Bowser';

Query OK, 1 row affected (0.01 sec)
Rows matched 1  Changed 1  Warnings 0

mysql SELECT * FROM pet WHERE name = 'Bowser';

+--------+-------+---------+------+------------+------------+
 name    owner  species  sex   birth       death      
+--------+-------+---------+------+------------+------------+
 Bowser  Diane  dog      m     1989-08-31  1995-07-29 
+--------+-------+---------+------+------------+------------+
1 row in set (0.00 sec)

mysql SELECT * FROM pet WHERE birth = '1998-1-1';

+----------+-------+---------+------+------------+-------+
 name      owner  species  sex   birth       death 
+----------+-------+---------+------+------------+-------+
 Chirpy    Gwen   bird     f     1998-09-11  NULL  
 Puffball  Diane  hamster  f     1999-03-30  NULL  
+----------+-------+---------+------+------------+-------+
2 rows in set (0.01 sec)
```

```sql
mysql SELECT * FROM pet WHERE species = 'dog' AND sex = 'f';

+-------+--------+---------+------+------------+-------+
 name   owner   species  sex   birth       death 
+-------+--------+---------+------+------------+-------+
 Buffy  Harold  dog      f     1989-05-13  NULL  
+-------+--------+---------+------+------------+-------+
1 row in set (0.00 sec)
```

```sql
mysql SELECT  FROM pet WHERE species = 'snake' OR species = 'bird';

+----------+-------+---------+------+------------+-------+
 name      owner  species  sex   birth       death 
+----------+-------+---------+------+------------+-------+
 Chirpy    Gwen   bird     f     1998-09-11  NULL  
 Whistler  Gwen   bird     NULL  1997-12-09  NULL  
 Slim      Benny  snake    m     1996-04-29  NULL  
+----------+-------+---------+------+------------+-------+
3 rows in set (0.00 sec)
```

```sql
mysql SELECT  FROM pet WHERE (species='cat' AND sex='m')
    - OR (species='dog' AND sex='f');

+-------+--------+---------+------+------------+-------+
 name   owner   species  sex   birth       death 
+-------+--------+---------+------+------------+-------+
 Claws  Gwen    cat      m     1994-03-17  NULL  
 Buffy  Harold  dog      f     1989-05-13  NULL  
+-------+--------+---------+------+------------+-------+
2 rows in set (0.00 sec)
```

```sql
mysql SELECT name, birth FROM pet;

+----------+------------+
 name      birth      
+----------+------------+
 Fluffy    1993-02-04 
 Claws     1994-03-17 
 Buffy     1989-05-13 
 Fang      1990-08-27 
 Bowser    1989-08-31 
 Chirpy    1998-09-11 
 Whistler  1997-12-09 
 Slim      1996-04-29 
 Puffball  1999-03-30 
+----------+------------+
9 rows in set (0.00 sec)
```

```sql
mysql SELECT owner FROM pet;

+--------+
 owner  
+--------+
 Harold 
 Gwen   
 Harold 
 Benny  
 Diane  
 Gwen   
 Gwen   
 Benny  
 Diane  
+--------+
9 rows in set (0.00 sec)
```

```sql
mysql SELECT DISTINCT owner FROM pet;

+--------+
 owner  
+--------+
 Harold 
 Gwen   
 Benny  
 Diane  
+--------+
4 rows in set (0.00 sec)
```

```sql
mysql SELECT name, species, birth FROM pet
    - WHERE species='dog' OR species='cat';

+--------+---------+------------+
 name    species  birth      
+--------+---------+------------+
 Fluffy  cat      1993-02-04 
 Claws   cat      1994-03-17 
 Buffy   dog      1989-05-13 
 Fang    dog      1990-08-27 
 Bowser  dog      1989-08-31 
+--------+---------+------------+
5 rows in set (0.00 sec)
```

```sql
mysql SELECT name, birth FROM pet ORDER BY birth;

+----------+------------+
 name      birth      
+----------+------------+
 Buffy     1989-05-13 
 Bowser    1989-08-31 
 Fang      1990-08-27 
 Fluffy    1993-02-04 
 Claws     1994-03-17 
 Slim      1996-04-29 
 Whistler  1997-12-09 
 Chirpy    1998-09-11 
 Puffball  1999-03-30 
+----------+------------+
9 rows in set (0.00 sec)
```

```sql
mysql SELECT name, birth FROM pet ORDER BY birth DESC;

+----------+------------+
 name      birth      
+----------+------------+
 Puffball  1999-03-30 
 Chirpy    1998-09-11 
 Whistler  1997-12-09 
 Slim      1996-04-29 
 Claws     1994-03-17 
 Fluffy    1993-02-04 
 Fang      1990-08-27 
 Bowser    1989-08-31 
 Buffy     1989-05-13 
+----------+------------+
9 rows in set (0.00 sec)
```

```sql
mysql SELECT name, species, birth FROM pet
    - ORDER BY species, birth DESC;

+----------+---------+------------+
 name      species  birth      
+----------+---------+------------+
 Chirpy    bird     1998-09-11 
 Whistler  bird     1997-12-09 
 Claws     cat      1994-03-17 
 Fluffy    cat      1993-02-04 
 Fang      dog      1990-08-27 
 Bowser    dog      1989-08-31 
 Buffy     dog      1989-05-13 
 Puffball  hamster  1999-03-30 
 Slim      snake    1996-04-29 
+----------+---------+------------+
9 rows in set (0.01 sec)
```

```sql
mysql SELECT name, birth, CURDATE(),
    - TIMESTAMPDIFF(YEAR, birth, CURDATE()) AS age
    - FROM pet;

+----------+------------+------------+------+
 name      birth       CURDATE()   age  
+----------+------------+------------+------+
 Fluffy    1993-02-04  2022-04-22    29 
 Claws     1994-03-17  2022-04-22    28 
 Buffy     1989-05-13  2022-04-22    32 
 Fang      1990-08-27  2022-04-22    31 
 Bowser    1989-08-31  2022-04-22    32 
 Chirpy    1998-09-11  2022-04-22    23 
 Whistler  1997-12-09  2022-04-22    24 
 Slim      1996-04-29  2022-04-22    25 
 Puffball  1999-03-30  2022-04-22    23 
+----------+------------+------------+------+
9 rows in set (0.00 sec)
```

```sql
mysql SELECT name, birth, CURDATE(),
    - TIMESTAMPDIFF(YEAR, birth, CURDATE()) AS age
    - FROM pet ORDER BY age;

+----------+------------+------------+------+
 name      birth       CURDATE()   age  
+----------+------------+------------+------+
 Chirpy    1998-09-11  2022-04-22    23 
 Puffball  1999-03-30  2022-04-22    23 
 Whistler  1997-12-09  2022-04-22    24 
 Slim      1996-04-29  2022-04-22    25 
 Claws     1994-03-17  2022-04-22    28 
 Fluffy    1993-02-04  2022-04-22    29 
 Fang      1990-08-27  2022-04-22    31 
 Buffy     1989-05-13  2022-04-22    32 
 Bowser    1989-08-31  2022-04-22    32 
+----------+------------+------------+------+
9 rows in set (0.00 sec)
```

```sql
mysql SELECT name, birth, death,
    - TIMESTAMPDIFF(YEAR, birth, death) AS age
    - FROM pet WHERE death IS NOT NULL ORDER BY age;

+--------+------------+------------+------+
 name    birth       death       age  
+--------+------------+------------+------+
 Bowser  1989-08-31  1995-07-29     5 
+--------+------------+------------+------+
1 row in set (0.00 sec)
```

```sql
mysql SELECT name, birth, MONTH(birth) FROM pet;

+----------+------------+--------------+
 name      birth       MONTH(birth) 
+----------+------------+--------------+
 Fluffy    1993-02-04             2 
 Claws     1994-03-17             3 
 Buffy     1989-05-13             5 
 Fang      1990-08-27             8 
 Bowser    1989-08-31             8 
 Chirpy    1998-09-11             9 
 Whistler  1997-12-09            12 
 Slim      1996-04-29             4 
 Puffball  1999-03-30             3 
+----------+------------+--------------+
9 rows in set (0.01 sec)

mysql SELECT name, birth, MONTH(birth) FROM pet WHERE MONTH(birth)=5;

+-------+------------+--------------+
 name   birth       MONTH(birth) 
+-------+------------+--------------+
 Buffy  1989-05-13             5 
+-------+------------+--------------+
1 row in set (0.00 sec)
```

```sql
mysql SELECT name, birth FROM pet
    - WHERE MONTH(birth) = MONTH(DATE_ADD(CURDATE(), INTERVAL 1 MONTH));

+-------+------------+
 name   birth      
+-------+------------+
 Buffy  1989-05-13 
+-------+------------+
1 row in set (0.01 sec)

mysql SELECT name, birth FROM pet
    - WHERE MONTH(birth) = MOD(MONTH(CURDATE()),12) + 1;

+-------+------------+
 name   birth      
+-------+------------+
 Buffy  1989-05-13 
+-------+------------+
1 row in set (0.00 sec)
```

```sql
mysql SELECT '2018-10-31' + INTERVAL 1 DAY;

+-------------------------------+
 '2018-10-31' + INTERVAL 1 DAY 
+-------------------------------+
 2018-11-01                    
+-------------------------------+
1 row in set (0.00 sec)

mysql SELECT '2018-10-32' + INTERVAL 1 DAY;

+-------------------------------+
 '2018-10-32' + INTERVAL 1 DAY 
+-------------------------------+
 NULL                          
+-------------------------------+
1 row in set, 1 warning (0.01 sec)

mysql SHOW WARNINGS;

+---------+------+----------------------------------------+
 Level    Code  Message                                
+---------+------+----------------------------------------+
 Warning  1292  Incorrect datetime value '2018-10-32' 
+---------+------+----------------------------------------+
1 row in set (0.00 sec)

mysql SELECT 1 IS NULL, 1 IS NOT NULL;

+-----------+---------------+
 1 IS NULL  1 IS NOT NULL 
+-----------+---------------+
         0              1 
+-----------+---------------+
1 row in set (0.00 sec)

mysql SELECT 0 IS NULL, 0 IS NOT NULL, '' IS NULL, '' IS NOT NULL;

+-----------+---------------+------------+----------------+
 0 IS NULL  0 IS NOT NULL  '' IS NULL  '' IS NOT NULL 
+-----------+---------------+------------+----------------+
         0              1           0               1 
+-----------+---------------+------------+----------------+
1 row in set (0.00 sec)
```

```sql
mysql SELECT  FROM pet WHERE name LIKE 'b%';
+--------+--------+---------+------+------------+------------+
 name    owner   species  sex   birth       death      
+--------+--------+---------+------+------------+------------+
 Buffy   Harold  dog      f     1989-05-13  NULL       
 Bowser  Diane   dog      m     1989-08-31  1995-07-29 
+--------+--------+---------+------+------------+------------+
2 rows in set (0.00 sec)

mysql SELECT  FROM pet WHERE name LIKE '%fy';
+--------+--------+---------+------+------------+-------+
 name    owner   species  sex   birth       death 
+--------+--------+---------+------+------------+-------+
 Fluffy  Harold  cat      f     1993-02-04  NULL  
 Buffy   Harold  dog      f     1989-05-13  NULL  
+--------+--------+---------+------+------------+-------+
2 rows in set (0.00 sec)

mysql SELECT  FROM pet WHERE name LIKE '%w%';
+----------+-------+---------+------+------------+------------+
 name      owner  species  sex   birth       death      
+----------+-------+---------+------+------------+------------+
 Claws     Gwen   cat      m     1994-03-17  NULL       
 Bowser    Diane  dog      m     1989-08-31  1995-07-29 
 Whistler  Gwen   bird     NULL  1997-12-09  NULL       
+----------+-------+---------+------+------------+------------+
3 rows in set (0.00 sec)

mysql SELECT  FROM pet WHERE name LIKE '_____';
+-------+--------+---------+------+------------+-------+
 name   owner   species  sex   birth       death 
+-------+--------+---------+------+------------+-------+
 Claws  Gwen    cat      m     1994-03-17  NULL  
 Buffy  Harold  dog      f     1989-05-13  NULL  
+-------+--------+---------+------+------------+-------+
2 rows in set (0.00 sec)
```

```sql
mysql SELECT  FROM pet WHERE REGEXP_LIKE(name, '^b');
+--------+--------+---------+------+------------+------------+
 name    owner   species  sex   birth       death      
+--------+--------+---------+------+------------+------------+
 Buffy   Harold  dog      f     1989-05-13  NULL       
 Bowser  Diane   dog      m     1989-08-31  1995-07-29 
+--------+--------+---------+------+------------+------------+
2 rows in set (0.02 sec)

mysql SELECT  FROM pet WHERE REGEXP_LIKE(name, 'fy$');
+--------+--------+---------+------+------------+-------+
 name    owner   species  sex   birth       death 
+--------+--------+---------+------+------------+-------+
 Fluffy  Harold  cat      f     1993-02-04  NULL  
 Buffy   Harold  dog      f     1989-05-13  NULL  
+--------+--------+---------+------+------------+-------+
2 rows in set (0.00 sec)

mysql SELECT  FROM pet WHERE REGEXP_LIKE(name, 'w');
+----------+-------+---------+------+------------+------------+
 name      owner  species  sex   birth       death      
+----------+-------+---------+------+------------+------------+
 Claws     Gwen   cat      m     1994-03-17  NULL       
 Bowser    Diane  dog      m     1989-08-31  1995-07-29 
 Whistler  Gwen   bird     NULL  1997-12-09  NULL       
+----------+-------+---------+------+------------+------------+
3 rows in set (0.00 sec)

mysql SELECT  FROM pet WHERE REGEXP_LIKE(name, '^.....$');
+-------+--------+---------+------+------------+-------+
 name   owner   species  sex   birth       death 
+-------+--------+---------+------+------------+-------+
 Claws  Gwen    cat      m     1994-03-17  NULL  
 Buffy  Harold  dog      f     1989-05-13  NULL  
+-------+--------+---------+------+------------+-------+
2 rows in set (0.00 sec)

mysql SELECT  FROM pet WHERE REGEXP_LIKE(name, '^.{5}$');
+-------+--------+---------+------+------------+-------+
 name   owner   species  sex   birth       death 
+-------+--------+---------+------+------------+-------+
 Claws  Gwen    cat      m     1994-03-17  NULL  
 Buffy  Harold  dog      f     1989-05-13  NULL  
+-------+--------+---------+------+------------+-------+
2 rows in set (0.00 sec)
```

```sql
mysql SELECT COUNT() FROM pet;
+----------+
 COUNT() 
+----------+
        9 
+----------+
1 row in set (0.01 sec)

mysql SELECT owner, COUNT() FROM pet GROUP BY owner;
+--------+----------+
 owner   COUNT() 
+--------+----------+
 Harold         2 
 Gwen           3 
 Benny          2 
 Diane          2 
+--------+----------+
4 rows in set (0.00 sec)

mysql SELECT species, COUNT() FROM pet GROUP BY species;
+---------+----------+
 species  COUNT() 
+---------+----------+
 cat             2 
 dog             3 
 bird            2 
 snake           1 
 hamster         1 
+---------+----------+
5 rows in set (0.00 sec)

mysql SELECT sex, COUNT() FROM pet GROUP BY sex;
+------+----------+
 sex   COUNT() 
+------+----------+
 f            4 
 m            4 
 NULL         1 
+------+----------+
3 rows in set (0.00 sec)

mysql SELECT species, sex, COUNT() FROM pet GROUP BY species, sex;
+---------+------+----------+
 species  sex   COUNT() 
+---------+------+----------+
 cat      f            1 
 cat      m            1 
 dog      f            1 
 dog      m            2 
 bird     f            1 
 bird     NULL         1 
 snake    m            1 
 hamster  f            1 
+---------+------+----------+
8 rows in set (0.00 sec)
```

```sql
mysql SELECT species, sex, COUNT() FROM pet
    - WHERE species = 'dog' OR species = 'cat'
    - GROUP BY species, sex;
+---------+------+----------+
 species  sex   COUNT() 
+---------+------+----------+
 cat      f            1 
 cat      m            1 
 dog      f            1 
 dog      m            2 
+---------+------+----------+
4 rows in set (0.00 sec)
```

```sql
mysql SELECT species, sex, COUNT() FROM pet
    - WHERE sex IS NOT NULL
    - GROUP BY species, sex;
+---------+------+----------+
 species  sex   COUNT() 
+---------+------+----------+
 cat      f            1 
 cat      m            1 
 dog      f            1 
 dog      m            2 
 bird     f            1 
 snake    m            1 
 hamster  f            1 
+---------+------+----------+
7 rows in set (0.00 sec)
```

```sql
mysql SELECT owner, COUNT() FROM pet;
+--------+----------+
 owner   COUNT() 
+--------+----------+
 Harold         9 
+--------+----------+
1 row in set (0.00 sec)
```

```sql
mysql CREATE TABLE event (
    - name VARCHAR(20),
    - date DATE,
    - type VARCHAR(15),
    - remark VARCHAR(255)
    - );
Query OK, 0 rows affected (0.05 sec)

mysql LOAD DATA LOCAL INFILE "CnotesSQLmenagerie-dbevent.txt INTO TABLE event;
Query OK, 10 rows affected, 2 warnings (0.01 sec)
Records 10  Deleted 0  Skipped 0  Warnings 2
```

```sql
mysql SELECT * FROM event;
+----------+------------+----------+-----------------------------+
 name      date        type      remark                      
+----------+------------+----------+-----------------------------+
 Fluffy    1995-05-15  litter    4 kittens, 3 female, 1 male 
 Buffy     1993-06-23  litter    5 puppies, 2 female, 3 male 
 Buffy     1994-06-19  litter    3 puppies, 3 female         
 Chirpy    1999-03-21  vet       needed beak straightened    
 Slim      1997-08-03  vet       broken rib                  
 Bowser    1991-10-12  kennel    NULL                        
 Fang      1991-10-12  kennel    NULL                        
 Fang      1998-08-28  birthday  Gave him a new chew toy     
 Claws     1998-03-17  birthday  Gave him a new flea collar  
 Whistler  1998-12-09  birthday  First birthday              
+----------+------------+----------+-----------------------------+
10 rows in set (0.00 sec)
```

```sql
mysql DESCRIBE pet;
+---------+-------------+------+-----+---------+-------+
 Field    Type         Null  Key  Default  Extra 
+---------+-------------+------+-----+---------+-------+
 name     varchar(20)  YES        NULL           
 owner    varchar(20)  YES        NULL           
 species  varchar(20)  YES        NULL           
 sex      char(1)      YES        NULL           
 birth    date         YES        NULL           
 death    date         YES        NULL           
+---------+-------------+------+-----+---------+-------+
6 rows in set (0.01 sec)
```

```sql
mysql SELECT pet.name,
    - TIMESTAMPDIFF(YEAR, birth, date) AS age,
    - remark
    - FROM pet INNER JOIN event
    - ON pet.name = event.name
    - WHERE event.type = 'litter';
+--------+------+-----------------------------+
 name    age   remark                      
+--------+------+-----------------------------+
 Fluffy     2  4 kittens, 3 female, 1 male 
 Buffy      5  3 puppies, 3 female         
 Buffy      4  5 puppies, 2 female, 3 male 
+--------+------+-----------------------------+
3 rows in set (0.01 sec)
```

```sql

```
