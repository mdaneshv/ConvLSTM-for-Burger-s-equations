use classicmodels;
SELECT 
    *
FROM
    customers;
SELECT 
    phone
FROM
    customers
WHERE
    phone LIKE ('508%');
insert into customers (customerNumber, customerName)
values (99999, 'Stephan');
SELECT 
    contactFirstName, COUNT(contactFirstName)
FROM
    customers
GROUP BY contactLastName;
SELECT 
    *
FROM
    payments;
SELECT 
    customerNumber, SUM(amount) AS summation
FROM
    payments
WHERE
    paymentDate > '2004-01-01'
GROUP BY customerNumber
HAVING summation > 1000
ORDER BY summation;
CREATE TABLE dup_orders (
    `orderNumber` INT NOT NULL,
    `orderDate` DATE NOT NULL,
    `requiredDate` DATE NOT NULL,
    `shippedDate` DATE DEFAULT NULL,
    `status` VARCHAR(15) NOT NULL,
    `comments` TEXT,
    `customerNumber` INT NOT NULL
);
  insert into dup_orders
  select * from orders;
SELECT 
    *
FROM
    dup_orders;
  insert into dup_orders (orderNumber, orderDate)
  values (100, '3000-01-01'),
  (300 ,'9000-01-01');
SELECT 
    *
FROM
    dup_orders
ORDER BY orderNumber ASC;

Commit;
use classicmodels;

delete from dup_orders where orderNumber=100;

select * from dup_orders order by orderNumber asc;
rollback;
select * from dup_orders;
commit;

delete from dup_orders where orderNumber=300;

select* from dup_orders order by orderNumber asc;

rollback;
select * from dup_orders order by orderNumber;

commit;

update dup_orders set orderDate='9999-01-01' where orderNumber=10100;

rollback;

select * from dup_orders;

delete from dup_orders where orderNumber=10100;

select * from dup_orders order by orderNumber asc;
rollback;



