-- use database
Use data;


-- Create a table
DROP TABLE IF EXISTS orders;
CREATE TABLE IF NOT EXISTS orders (
    `orderNumber` INT NOT NULL,
    `orderDate` DATE NULL,
    `requiredDate` DATE NULL,
    `shippedDate` DATE DEFAULT NULL,
    `status` VARCHAR(15) NULL,
    `comments` TEXT,
    `customerNumber` INT NOT NULL,
    `customerName` VARCHAR(30) NOT NULL,
    `customerLocation` VARCHAR(30) NULL
);


-- Insert into the table from a csv or txt file. Use FIRSTROW=2 to skip the column headers.
LOAD DATA INFILE 'C:\orders.csv'
INTO TABLE orders
FIELDS TERMINATED BY ',';
   
   
-- Insert into the table using select statement and another table
INSERT INTO orders
  SELECT * FROM data.orders;
  
  
-- Or insert manually
INSERT INTO orders (orderNumber, orderDate)
  VALUES (1, '2019-01-01'),
  (2 ,'2020-01-01');
  

--  Select all data in a ascending order of ordernumbers
SELECT 
    *
FROM
    orders
ORDER BY orderNumber ASC;


-- A different table named customer
SELECT 
    *
FROM
    customers;
    

-- Extract all phone numbers starting with 508  
SELECT 
    phone
FROM
    customers
WHERE
    phone LIKE ('508%');


-- Select with aliases 
SELECT 
    contactFirstName, COUNT(contactFirstName) AS FirstNameCount
FROM
    customers
GROUP BY contactFirstName;


--  Return a table with customerNumber and summation columns including information
--  about customers who have paid more than $1000 after 2020-01-01.
SELECT 
    customerNumber, SUM(amount) AS summation
FROM
    payments
WHERE
    paymentDate > '2020-01-01'
GROUP BY customerNumber
HAVING summation > 1000
ORDER BY summation;


COMMIT;-- A transaction control language used to save the changes permanently.
DELETE FROM orders 
WHERE
    orderNumber = 2;
SELECT 
    *
FROM
    orders
ORDER BY orderNumber ASC;
-- Rollback changes everything back to the last commit.
ROLLBACK;
SELECT 
    *
FROM
    orders;



COMMIT; -- Changes are saved and this transaction can not be undone.
UPDATE orders SET orderDate='9999-01-01' WHERE orderNumber=1;
ROLLBACK; -- Changes back to the last commit.

