-- schema.sql
CREATE TABLE energy_consumption (
    household_id VARCHAR(50),
    timestamp TIMESTAMP,
    energy_consumption FLOAT
);
