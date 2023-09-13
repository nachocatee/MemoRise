package com.tjjhtjh.memorise;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.security.servlet.SecurityAutoConfiguration;

@SpringBootApplication(exclude = SecurityAutoConfiguration.class)
public class MemoriseApplication {

	public static void main(String[] args) {
		SpringApplication.run(MemoriseApplication.class, args);
	}

}
