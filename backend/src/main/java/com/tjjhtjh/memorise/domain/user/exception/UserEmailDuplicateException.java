package com.tjjhtjh.memorise.domain.user.exception;

public class UserEmailDuplicateException extends RuntimeException {
    public UserEmailDuplicateException(String message) {
        super(message);
    }

    public UserEmailDuplicateException(String message, Throwable cause) {
        super(message, cause);
    }
}
