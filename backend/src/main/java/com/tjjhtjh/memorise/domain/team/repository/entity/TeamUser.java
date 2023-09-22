package com.tjjhtjh.memorise.domain.team.repository.entity;

import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.DynamicInsert;
import org.hibernate.annotations.DynamicUpdate;
import org.springframework.data.relational.core.mapping.Table;

import java.io.Serializable;

@DynamicUpdate
@DynamicInsert
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@AllArgsConstructor
@Builder
@Entity
@Table(name = "team_user")
public class TeamUser implements Serializable {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "team_user_seq")
    private Long teamUserSeq;
    //    @JsonBackReference(value = "group-user-middle")
    @ManyToOne(targetEntity = Team.class, fetch = FetchType.LAZY)
    @JoinColumn(name = "team_seq", referencedColumnName = "team_seq")
    private Team team;
    //    @JsonBackReference(value = "user-group-middle")
    @ManyToOne(targetEntity = User.class, fetch = FetchType.LAZY)
    @JoinColumn(name = "user_seq", referencedColumnName = "user_seq")
    private User user;
}
