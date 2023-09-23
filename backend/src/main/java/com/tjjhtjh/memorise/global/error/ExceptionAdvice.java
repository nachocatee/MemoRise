package com.tjjhtjh.memorise.global.error;

import com.tjjhtjh.memorise.domain.team.exception.NoAuthorityException;
import com.tjjhtjh.memorise.domain.team.exception.NoTeamException;
import com.tjjhtjh.memorise.domain.team.exception.NotMemberOfGroup;
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
        ErrorResponse errorResponse = new ErrorResponse("이미 존재하는 이메일입니다.");
        return ResponseEntity.ok(errorResponse);
    }

    @ExceptionHandler(NoUserException.class)
    public ResponseEntity<ErrorResponse> noUser() {
        ErrorResponse errorResponse = new ErrorResponse("유저가 존재하지 않습니다.");
        return ResponseEntity.ok(errorResponse);
    }

    @ExceptionHandler(NoTeamException.class)
    public ResponseEntity<ErrorResponse> noTeam() {
        ErrorResponse errorResponse = new ErrorResponse("그룹이 존재하지 않습니다.");
        return ResponseEntity.ok(errorResponse);
    }

    @ExceptionHandler(NotMemberOfGroup.class)
    public ResponseEntity<ErrorResponse> notGroupMember() {
        ErrorResponse errorResponse = new ErrorResponse("그룹에 속해있지 않습니다.");
        return ResponseEntity.ok(errorResponse);
    }

    @ExceptionHandler(NoAuthorityException.class)
    public ResponseEntity<ErrorResponse> noAuthority() {
        ErrorResponse errorResponse = new ErrorResponse("권한이 없습니다.");
        return ResponseEntity.ok(errorResponse);
    }
}
