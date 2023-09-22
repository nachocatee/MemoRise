package com.tjjhtjh.memorise.domain.user.service;

import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import com.tjjhtjh.memorise.domain.user.service.dto.request.JoinRequest;
import com.tjjhtjh.memorise.domain.user.service.dto.request.UpdateUserInfoRequest;
import com.tjjhtjh.memorise.domain.user.service.dto.response.UserListResponse;

import java.util.List;

public interface UserService {

    User join(JoinRequest joinRequest);
    void updateUserInfo(Long userSeq, UpdateUserInfoRequest updateUserInfoRequest);
    User getUserInfo(Long userSeq);
    UserListResponse getUserList(String nickname);

}
