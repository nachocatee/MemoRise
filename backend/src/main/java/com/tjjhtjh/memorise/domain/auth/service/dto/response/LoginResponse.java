package com.tjjhtjh.memorise.domain.auth.service.dto.response;

import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import lombok.Getter;
import lombok.Setter;

@Setter
@Getter
public class LoginResponse {
    boolean isSuccess;
    Long userSeq;
    String email;
    String nickName;
    String profile;

    public LoginResponse(boolean isSuccess, User user) {
        this.isSuccess = isSuccess;
        this.userSeq = user.getUserSeq();
        this.email = user.getEmail();
        this.nickName = user.getNickname();
        this.profile = user.getProfile();
    }
}
