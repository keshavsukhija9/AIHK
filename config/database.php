<?php
/**
 * Database Configuration for TravelX Application
 * Centralized database connection management
 */

class DatabaseConfig {
    private static $instance = null;
    private $connection = null;
    
    // Database configuration constants
    private const DB_HOST = "localhost";
    private const DB_USERNAME = "root";
    private const DB_PASSWORD = "8922095859";
    private const DB_NAME = "travelx";
    private const DB_CHARSET = "utf8mb4";
    
    private function __construct() {
        try {
            $this->connection = new mysqli(
                self::DB_HOST,
                self::DB_USERNAME,
                self::DB_PASSWORD,
                self::DB_NAME
            );
            
            // Set charset
            $this->connection->set_charset(self::DB_CHARSET);
            
            // Check connection
            if ($this->connection->connect_error) {
                throw new Exception("Database connection failed: " . $this->connection->connect_error);
            }
            
            // Set timezone
            $this->connection->query("SET time_zone = '+05:30'");
            
        } catch (Exception $e) {
            error_log("Database connection error: " . $e->getMessage());
            throw $e;
        }
    }
    
    public static function getInstance() {
        if (self::$instance === null) {
            self::$instance = new self();
        }
        return self::$instance;
    }
    
    public function getConnection() {
        // Ping to check if connection is alive
        if (!$this->connection->ping()) {
            // Reconnect if connection was lost
            $this->__construct();
        }
        return $this->connection;
    }
    
    public function closeConnection() {
        if ($this->connection && !$this->connection->connect_error) {
            $this->connection->close();
        }
        self::$instance = null;
    }
    
    /**
     * Prepare and execute a statement with error handling
     */
    public function prepareAndExecute($query, $types = '', $params = []) {
        try {
            $stmt = $this->connection->prepare($query);
            if (!$stmt) {
                throw new Exception("Prepare failed: " . $this->connection->error);
            }
            
            if (!empty($params)) {
                $stmt->bind_param($types, ...$params);
            }
            
            $stmt->execute();
            return $stmt;
            
        } catch (Exception $e) {
            error_log("Database query error: " . $e->getMessage());
            throw $e;
        }
    }
    
    /**
     * Get images for a service
     */
    public function getServiceImages($serviceType, $serviceId) {
        $query = "SELECT image_path, is_main FROM service_images WHERE service_type = ? AND service_id = ? ORDER BY is_main DESC, image_id ASC";
        $stmt = $this->prepareAndExecute($query, 'si', [$serviceType, $serviceId]);
        
        $images = [];
        $result = $stmt->get_result();
        while ($row = $result->fetch_assoc()) {
            $images[] = $row;
        }
        $stmt->close();
        
        return $images;
    }
    
    // Prevent cloning and unserialization
    private function __clone() {}
    public function __wakeup() {}
}

// Global helper function for easy access
function getDBConnection() {
    return DatabaseConfig::getInstance()->getConnection();
}
?>