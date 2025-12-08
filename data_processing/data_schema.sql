DROP TABLE IF EXISTS object;
DROP TABLE IF EXISTS data_type;
DROP TABLE IF EXISTS data_value;
DROP TABLE IF EXISTS object_linkage;

-- Creating table to store the generators, objects etc.
CREATE TABLE object (
    resource_id INT PRIMARY KEY,
    resource_name TEXT NOT NULL,
    fuel_type TEXT NOT NULL,
    qse TEXT NOT NULL,
    dme TEXT NOT NULL
)

-
CREATE TABLE data_value(
    resource_id FOREIGN KEY REFERENCES object(resource_id),
    data_structure_type NOT NULL REFERENCES data_type(name),
    value any NULL,
    name any NULL
)

CREATE TABLE data_type(
    name TEXT NOT NULL PRIMARY KEY,
    description TEXT NULL,
    validation_query TEXT
)

-- For linking SCED and price data objects etc.
CREATE TABLE object_linkage(
    FOREIGN KEY (resource_id) REFERENCES object(resource_id),
    FOREIGN KEY (resource_id) REFERENCES object(resource_id)
)

-- Pre-populate data_structure table, keep this rigid
INSERT INTO data_type (name, validation_query) VALUES
('integer', 'cast(? as integer) IS NOT NULL'),
('float', 'cast(? as float) IS NOT NULL'),
('text', 'cast(? as text) IS NOT NULL'),
('date', 'cast(? as date) IS NOT NULL'),
('datetime', 'cast(? as datetime) IS NOT NULL'),
('timeseries', '? IS NULL'),
('curve', '? IS NULL');