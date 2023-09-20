package com.tjjhtjh.memorise.global.error;

import com.tjjhtjh.memorise.domain.user.exception.NoUserException;
import com.tjjhtjh.memorise.domain.user.exception.UserEmailDuplicateException;
import com.tjjhtjh.memorise.global.error.dto.ErrorResponse;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
public class ExceptionAdvice {

    @ExceptionHandler(UserEmailDuplicateException.class)
    public ResponseEntity<ErrorResponse> handleException() {
        ErrorResponse errorResponse = new ErrorResponse("회원가입 정보가 잘못되었습니다.");
        return ResponseEntity.ok(errorResponse);
    }

    @ExceptionHandler(NoUserException.class)
    public ResponseEntity<ErrorResponse> noUser() {
        ErrorResponse errorResponse = new ErrorResponse("유저가 존재하지 않습니다.");
        return ResponseEntity.ok(errorResponse);
    }
}
